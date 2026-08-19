"""
CASO 3 - DIABETES  |  REGRESION POLINOMIAL
Universidad Andina del Cusco - Inteligencia Artificial

Progresion cuantitativa de la enfermedad un ano despues del inicio.

Fase A: analisis exploratorio y preprocesamiento.
Fase B: regresion polinomial (grados 1 a 3, con y sin regularizacion Ridge)
        y analisis del sobreajuste.
Fase C: exporta model.js para el aplicativo web.

Ejecutar:  python train.py
Genera:    figuras/*.png, model.js, resultados.json
"""

import json
import os
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

RANDOM_STATE = 42
CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
FIGDIR = "figuras"
os.makedirs(FIGDIR, exist_ok=True)
sns.set_theme(style="whitegrid")

resultados = {}


def titulo(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# =============================================================================
# FASE A - ANALISIS EXPLORATORIO Y PREPROCESAMIENTO
# =============================================================================
titulo("FASE A.0 - CARGA DEL DATASET")

# scaled=False devuelve las unidades clinicas reales, necesarias para aplicar
# StandardScaler (Fase A) y para pedir valores al usuario en el aplicativo (Fase C).
X, y = load_diabetes(return_X_y=True, as_frame=True, scaled=False)

DESCRIPCION = {
    "age": "Edad (anos)",
    "sex": "Sexo (0 / 1)",
    "bmi": "Indice de masa corporal (kg/m2)",
    "bp": "Presion arterial media (mm Hg)",
    "s1": "Colesterol total - tc (mg/dL)",
    "s2": "Lipoproteinas de baja densidad - ldl (mg/dL)",
    "s3": "Lipoproteinas de alta densidad - hdl (mg/dL)",
    "s4": "Razon colesterol total / HDL - tch",
    "s5": "Trigliceridos sericos, posiblemente en log - ltg",
    "s6": "Glucosa en sangre - glu (mg/dL)",
}

print(f"Muestras: {X.shape[0]}   Variables: {X.shape[1]}")
print(f"Target: progresion de la enfermedad. Rango {y.min():.0f} - {y.max():.0f}, "
      f"media {y.mean():.2f}, asimetria {y.skew():.3f}")

X["sex"] = (X["sex"] - 1).astype(float)   # unica categorica: 1/2 -> 0/1

titulo("FASE A.1 - TIPOS DE DATOS, NULOS Y DUPLICADOS")
print(pd.DataFrame({
    "dtype": X.dtypes.astype(str), "nulos": X.isna().sum(),
    "unicos": X.nunique(), "descripcion": pd.Series(DESCRIPCION),
}).to_string())
print(f"\nFilas duplicadas: {X.duplicated().sum()}")
print(f"Nulos totales: {int(X.isna().sum().sum())}  ->  no se requiere imputacion")

resultados["fase_a"] = {
    "n_muestras": int(X.shape[0]), "n_variables": int(X.shape[1]),
    "nulos": int(X.isna().sum().sum()), "duplicados": int(X.duplicated().sum()),
    "target_min": float(y.min()), "target_max": float(y.max()),
    "target_media": float(y.mean()), "target_asimetria": float(y.skew()),
}

titulo("FASE A.2 - OUTLIERS Y MULTICOLINEALIDAD")
outliers = {}
for col in X.columns:
    q1, q3 = X[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n = int(((X[col] < q1 - 1.5*iqr) | (X[col] > q3 + 1.5*iqr)).sum())
    outliers[col] = n
print("Outliers por IQR:", {k: v for k, v in outliers.items() if v})
print("Decision: se conservan (valores clinicos plausibles).")

corr = X.corr()
Z = StandardScaler().fit_transform(X)
vif = {}
for j, name in enumerate(X.columns):
    otras = [k for k in range(Z.shape[1]) if k != j]
    r2 = LinearRegression().fit(Z[:, otras], Z[:, j]).score(Z[:, otras], Z[:, j])
    vif[name] = float(1 / (1 - r2))
print("\nVIF:")
for k, v in sorted(vif.items(), key=lambda p: -p[1]):
    print(f"  {k:4s} {v:7.2f}{'  <-- SEVERA' if v > 10 else ''}")
print("\n  Importante para esta metodologia: la expansion polinomial se construye")
print("  sobre variables YA colineales (s4 = s1/s3, s2 componente de s1), de modo")
print("  que multiplica la redundancia en lugar de aportar informacion nueva.")
resultados["vif"] = vif
resultados["outliers"] = outliers

fig, ax = plt.subplots(figsize=(8.5, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, linewidths=.5, ax=ax, cbar_kws={"shrink": .8})
ax.set_title("Fase A - Matriz de correlacion de Pearson", fontsize=12)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/a_correlacion.png", dpi=120); plt.close(fig)

titulo("FASE A.3 - PARTICION 75/25 Y ESTANDARIZACION")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE)
print(f"Entrenamiento: {len(X_train)} muestras   Prueba: {len(X_test)} muestras")
print("\nEl orden del Pipeline es PolynomialFeatures -> StandardScaler -> modelo:")
print("la estandarizacion va DESPUES de la expansion, porque los terminos cuadraticos")
print("y cruzados tienen escalas radicalmente distintas de las variables originales.")
resultados["n_train"], resultados["n_test"] = len(X_train), len(X_test)


# =============================================================================
# FASE B - REGRESION POLINOMIAL
# =============================================================================
modelos = []


def evaluar(nombre, pipe):
    pipe.fit(X_train, y_train)
    p_tr, p_te = pipe.predict(X_train), pipe.predict(X_test)
    cv = cross_val_score(pipe, X_train, y_train, cv=CV, scoring="r2")
    m = {
        "nombre": nombre, "n_terminos": int(pipe[-1].coef_.size),
        "r2_train": float(r2_score(y_train, p_tr)),
        "r2_test": float(r2_score(y_test, p_te)),
        "r2_cv": float(cv.mean()), "r2_cv_std": float(cv.std()),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, p_te))),
        "mae_test": float(mean_absolute_error(y_test, p_te)),
    }
    m["brecha"] = m["r2_train"] - m["r2_cv"]
    print(f"  {nombre:34s} p={m['n_terminos']:4d}  R2tr={m['r2_train']:.4f}  "
          f"R2te={m['r2_test']:9.4f}  CV={m['r2_cv']:10.4f}  "
          f"brecha={m['brecha']:9.4f}  RMSE={m['rmse_test']:7.2f}")
    modelos.append(m)
    return m, pipe


titulo("FASE B.1 - TERMINOS GENERADOS POR GRADO")
print(f"{'Grado':>6}  {'Terminos':>9}  {'Muestras':>9}  {'Ratio n/p':>10}")
for grado in (1, 2, 3):
    nt = PolynomialFeatures(grado, include_bias=False).fit(X_train.iloc[:3]).n_output_features_
    print(f"{grado:>6}  {nt:>9}  {len(X_train):>9}  {len(X_train)/nt:>10.2f}")
print("\n  Con grado 3 hay practicamente un parametro por observacion: el modelo")
print("  puede reproducir el entrenamiento sin aprender nada generalizable.")

titulo("FASE B.2 - COMPARATIVA POR GRADO, CON Y SIN REGULARIZACION")
curva = []
for grado in (1, 2, 3):
    mo, _ = evaluar(f"Grado {grado} - OLS (sin regular.)", Pipeline([
        ("poly", PolynomialFeatures(degree=grado, include_bias=False)),
        ("sc", StandardScaler()), ("lr", LinearRegression())]))
    mr, pr = evaluar(f"Grado {grado} - Ridge (alpha por CV)", Pipeline([
        ("poly", PolynomialFeatures(degree=grado, include_bias=False)),
        ("sc", StandardScaler()),
        ("lr", RidgeCV(alphas=np.logspace(-2, 5, 71), cv=CV))]))
    curva.append({"grado": grado, "n_terminos": mo["n_terminos"],
                  "ols_train": mo["r2_train"], "ols_cv": mo["r2_cv"],
                  "ols_test": mo["r2_test"],
                  "ridge_train": mr["r2_train"], "ridge_cv": mr["r2_cv"],
                  "ridge_test": mr["r2_test"], "alpha": float(pr[-1].alpha_)})

print("\n  Grado 2 sin regularizar: el R2 de entrenamiento sube y el de validacion")
print("  cruzada se hunde. Grado 3 sin regularizar es una catastrofe (R2 negativo:")
print("  peor que predecir siempre la media). Sobreajuste de manual.")
print("\n  alpha elegido por CV en cada grado: " +
      ", ".join(f"gr.{c['grado']} = {c['alpha']:.2f}" for c in curva))
print("  Crece con el grado: a mas terminos, mas penalizacion hace falta.")
print("  Pero los tres grados convergen al mismo R2 de CV (~0.460), que es el del")
print("  modelo lineal: Ridge no extrae informacion de los terminos, los neutraliza.")
resultados["curva_polinomial"] = curva
resultados["comparativa"] = modelos

# --- curva de validacion
fig, ax = plt.subplots(figsize=(9, 5))
g = [c["grado"] for c in curva]
ax.plot(g, [c["ols_train"] for c in curva], "o-", c="#0d3c6c", label="OLS - entrenamiento")
ax.plot(g, [c["ols_cv"] for c in curva], "o--", c="#0d3c6c", alpha=.65, label="OLS - validacion cruzada")
ax.plot(g, [c["ridge_train"] for c in curva], "s-", c="#1d9e6f", label="Ridge - entrenamiento")
ax.plot(g, [c["ridge_cv"] for c in curva], "s--", c="#1d9e6f", alpha=.65, label="Ridge - validacion cruzada")

# El R2 de CV del grado 3 sin regularizar (-171) aplastaria la escala: se recorta
# el eje y se anota el valor fuera de rango.
ax.set_ylim(-0.35, 1.0)
ax.axhline(0, c="k", lw=.8)
ax.set_xticks(g); ax.set_xlabel("Grado del polinomio"); ax.set_ylabel("R2")
ax.set_title("Fase B - Curva de validacion: el sobreajuste crece con el grado")
for c in curva:
    ax.annotate(f"{c['n_terminos']} terminos", (c["grado"], c["ols_train"]),
                textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
for c in [c for c in curva if c["ols_cv"] < -0.35]:
    ax.annotate(f"R2 = {c['ols_cv']:.1f}\n(fuera de escala)", (c["grado"], -0.32),
                textcoords="offset points", xytext=(-14, 8), ha="right", fontsize=9,
                color="#d62828", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="#d62828", lw=1.4))
    ax.plot([c["grado"]], [-0.34], marker="v", ms=11, color="#d62828", clip_on=False)
ax.legend(loc="lower left", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/b_curva_validacion.png", dpi=120); plt.close(fig)

# --- brecha entrenamiento vs validacion
fig, ax = plt.subplots(figsize=(8, 4.5))
ancho = 0.35
xs = np.arange(len(curva))
ax.bar(xs - ancho/2, [c["ols_train"] - c["ols_cv"] for c in curva], ancho,
       color="#d62828", label="Sin regularizar")
ax.bar(xs + ancho/2, [c["ridge_train"] - c["ridge_cv"] for c in curva], ancho,
       color="#1d9e6f", label="Con Ridge")
ax.set_yscale("symlog", linthresh=0.1)
ax.set_xticks(xs); ax.set_xticklabels([f"Grado {c['grado']}\n({c['n_terminos']} term.)" for c in curva])
ax.set_ylabel("Brecha R2 entrenamiento - validacion (escala log)")
ax.set_title("Fase B - La regularizacion contiene el sobreajuste")
ax.legend()
for i, c in enumerate(curva):
    ax.annotate(f"{c['ols_train'] - c['ols_cv']:.2f}", (i - ancho/2, c["ols_train"] - c["ols_cv"]),
                ha="center", va="bottom", fontsize=8.5, color="#d62828", fontweight="bold")
fig.tight_layout(); fig.savefig(f"{FIGDIR}/b_brecha.png", dpi=120); plt.close(fig)

titulo("FASE B.3 - MODELO POLINOMIAL SELECCIONADO")
mejor = max(modelos, key=lambda m: m["r2_cv"])
lineal = next(m for m in modelos if m["nombre"].startswith("Grado 1 - OLS"))
print(f"  Mejor R2 de validacion cruzada : {mejor['nombre']} ({mejor['r2_cv']:.4f})")
print(f"  Regresion lineal (grado 1)     : {lineal['r2_cv']:.4f}")
print(f"  Diferencia {mejor['r2_cv'] - lineal['r2_cv']:+.4f}, frente a una desviacion "
      f"entre particiones de {mejor['r2_cv_std']:.3f}.")
print("\n  Los modelos son estadisticamente INDISTINGUIBLES. El polinomial de grado 2")
print("  con Ridge se exporta como representante de esta metodologia, pero por")
print("  parsimonia el modelo recomendado del Caso 3 es el lineal multiple:")
print("  10 terminos en lugar de 65 y coeficientes interpretables.")
resultados["modelo_exportado"] = "Grado 2 - Ridge (alpha por CV)"


# =============================================================================
# FASE C - EXPORTACION PARA EL APLICATIVO WEB
# =============================================================================
titulo("FASE C - EXPORTACION DEL MODELO")

pipe_final = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("sc", StandardScaler()),
    ("lr", RidgeCV(alphas=np.logspace(-2, 5, 71), cv=CV))]).fit(X_train, y_train)
m_final = next(m for m in modelos if m["nombre"] == "Grado 2 - Ridge (alpha por CV)")

TODAS = list(X.columns)
sc_ = pipe_final.named_steps["sc"]
lr_ = pipe_final[-1]

payload = {
    "caso": "Caso 3 - Diabetes | Regresion Polinomial",
    "metodologia": "Regresion Polinomial grado 2 con Ridge (alpha por CV)",
    "descripcion": DESCRIPCION,
    "variables": TODAS,
    "grado": 2,
    # Los exponentes de cada termino permiten a JavaScript reconstruir exactamente
    # la misma expansion que PolynomialFeatures de scikit-learn.
    "powers": pipe_final.named_steps["poly"].powers_.tolist(),
    "rangos": {c: [float(X[c].min()), float(X[c].max())] for c in TODAS},
    "medianas": {c: float(X[c].median()) for c in TODAS},
    "target": {"min": float(y.min()), "max": float(y.max()), "media": float(y.mean())},
    "n_train": len(X_train), "n_test": len(X_test),
    "mu": [float(v) for v in sc_.mean_],
    "sigma": [float(v) for v in sc_.scale_],
    "coef": [float(v) for v in lr_.coef_],
    "intercept": float(lr_.intercept_),
    "alpha": float(lr_.alpha_),
    "metricas": m_final,
    "comparativa": modelos,
    "curva": curva,
    "vif": vif,
}

with open("model.js", "w", encoding="utf-8") as f:
    f.write("// Generado por train.py - Caso 3 Diabetes | Regresion Polinomial\n")
    f.write("const MODELO_POLINOMIAL = " + json.dumps(payload, indent=2, ensure_ascii=False) + ";\n")

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

print(f"  model.js         -> grado 2, {len(payload['coef'])} terminos, alpha={lr_.alpha_:.2f}")
print("  resultados.json  -> metricas de las Fases A y B")
print(f"  {FIGDIR}/         -> 3 figuras")
print("\nListo.")
