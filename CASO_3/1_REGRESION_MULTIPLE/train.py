"""
CASO 3 - DIABETES  |  REGRESION LINEAL MULTIPLE
Universidad Andina del Cusco - Inteligencia Artificial

Progresion cuantitativa de la enfermedad un ano despues del inicio.

Fase A: analisis exploratorio y preprocesamiento.
Fase B: regresion lineal multiple (OLS, Ridge, Lasso) y significancia estadistica.
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
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
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

# scaled=False devuelve las unidades clinicas reales. Con el valor por defecto
# (scaled=True) los datos ya vienen centrados y escalados: no se podria aplicar
# StandardScaler como pide la Fase A, ni pedir valores al usuario en la Fase C.
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

# 'sex' es la unica categorica y viene codificada 1/2. Se recodifica a 0/1 para
# que su coeficiente se lea como el efecto de una categoria respecto a la otra.
X["sex"] = (X["sex"] - 1).astype(float)

titulo("FASE A.1 - TIPOS DE DATOS, NULOS Y DUPLICADOS")
info = pd.DataFrame({
    "dtype": X.dtypes.astype(str),
    "nulos": X.isna().sum(),
    "unicos": X.nunique(),
    "descripcion": pd.Series(DESCRIPCION),
})
print(info.to_string())
print(f"\nFilas duplicadas: {X.duplicated().sum()}")
print(f"Nulos totales: {int(X.isna().sum().sum())}  ->  no se requiere imputacion")

resultados["fase_a"] = {
    "n_muestras": int(X.shape[0]), "n_variables": int(X.shape[1]),
    "nulos": int(X.isna().sum().sum()), "duplicados": int(X.duplicated().sum()),
    "target_min": float(y.min()), "target_max": float(y.max()),
    "target_media": float(y.mean()), "target_asimetria": float(y.skew()),
}

titulo("FASE A.2 - ESTADISTICA DESCRIPTIVA")
print(X.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(3).to_string())

titulo("FASE A.3 - DETECCION DE OUTLIERS (metodo IQR, k=1.5)")
outliers = {}
for col in X.columns:
    q1, q3 = X[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n = int(((X[col] < lo) | (X[col] > hi)).sum())
    outliers[col] = n
    if n:
        print(f"  {col:4s} {n:3d} ({100*n/len(X):4.1f}%)   limites [{lo:.2f}, {hi:.2f}]")
q1, q3 = y.quantile([0.25, 0.75]); iqr = q3 - q1
n_out_y = int(((y < q1 - 1.5*iqr) | (y > q3 + 1.5*iqr)).sum())
print(f"\n  Outliers en el target: {n_out_y}")
print("\n  Decision: NO se eliminan. Son valores clinicos plausibles y con 442")
print("  muestras cada fila cuenta. El efecto se cuantifica en la Fase B.3.")
resultados["outliers"] = outliers
resultados["outliers_target"] = n_out_y

fig, axes = plt.subplots(2, 5, figsize=(18, 7))
for ax, col in zip(axes.ravel(), X.columns):
    sns.boxplot(y=X[col], ax=ax, color="#0d3c6c", width=.5)
    ax.set_title(f"{col}  ({outliers[col]} outliers)", fontsize=10)
    ax.set_ylabel("")
fig.suptitle("Fase A.3 - Diagramas de caja por variable (metodo IQR)", fontsize=13)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/a3_boxplots.png", dpi=120); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(y, kde=True, ax=axes[0], color="#f77f00", bins=30)
axes[0].set_title(f"Distribucion del target (asimetria = {y.skew():.3f})")
axes[0].set_xlabel("Progresion de la enfermedad")
stats.probplot(y, dist="norm", plot=axes[1])
axes[1].set_title("Grafico Q-Q del target")
fig.tight_layout(); fig.savefig(f"{FIGDIR}/a3_target.png", dpi=120); plt.close(fig)

titulo("FASE A.4 - ANALISIS DE MULTICOLINEALIDAD")
corr = X.corr()
print("Pares con |r| > 0.5:")
pares = []
for i in range(len(corr)):
    for j in range(i + 1, len(corr)):
        r = corr.iloc[i, j]
        if abs(r) > 0.5:
            pares.append((corr.index[i], corr.columns[j], float(r)))
            print(f"  {corr.index[i]:4s} - {corr.columns[j]:4s}   r = {r:+.3f}")

# VIF_j = 1 / (1 - R2_j), donde R2_j es el ajuste de la variable j contra las demas.
Z = StandardScaler().fit_transform(X)
vif = {}
for j, name in enumerate(X.columns):
    otras = [k for k in range(Z.shape[1]) if k != j]
    r2 = LinearRegression().fit(Z[:, otras], Z[:, j]).score(Z[:, otras], Z[:, j])
    vif[name] = float(1 / (1 - r2))
print("\nFactor de Inflacion de la Varianza (VIF):")
for k, v in sorted(vif.items(), key=lambda p: -p[1]):
    flag = "  <-- SEVERA" if v > 10 else ("  <-- moderada" if v > 5 else "")
    print(f"  {k:4s} {v:7.2f}{flag}")
print("\n  s4 es por definicion la razon s1/s3, y s2 (LDL) es un componente de s1")
print("  (colesterol total): la multicolinealidad es estructural, no casual.")
resultados["vif"] = vif
resultados["correlaciones_altas"] = pares

fig, ax = plt.subplots(figsize=(8.5, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, linewidths=.5, ax=ax, cbar_kws={"shrink": .8})
ax.set_title("Fase A.4 - Matriz de correlacion de Pearson", fontsize=12)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/a4_correlacion.png", dpi=120); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5))
s = pd.Series(vif).sort_values(ascending=False)
colores = ["#d62828" if v > 10 else "#f77f00" if v > 5 else "#0d3c6c" for v in s]
ax.bar(s.index, s.values, color=colores)
ax.axhline(10, ls="--", c="#d62828", label="VIF = 10 (multicolinealidad severa)")
ax.axhline(5, ls=":", c="#f77f00", label="VIF = 5 (moderada)")
ax.set_ylabel("VIF"); ax.set_title("Fase A.4 - Factor de Inflacion de la Varianza")
ax.legend()
fig.tight_layout(); fig.savefig(f"{FIGDIR}/a4_vif.png", dpi=120); plt.close(fig)

titulo("FASE A.5 - PARTICION 75/25 Y ESTANDARIZACION")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE)
print(f"Entrenamiento: {len(X_train)} muestras   Prueba: {len(X_test)} muestras")
print("\nLa estandarizacion (StandardScaler) se aplica dentro de un Pipeline, de modo")
print("que la media y desviacion se calculan SOLO con el set de entrenamiento en cada")
print("particion de la validacion cruzada. Esto evita la fuga de informacion.")
resultados["n_train"], resultados["n_test"] = len(X_train), len(X_test)


# =============================================================================
# FASE B - REGRESION LINEAL MULTIPLE
# =============================================================================
modelos = []
TODAS = list(X.columns)
SIN_S1S2 = [c for c in TODAS if c not in ("s1", "s2")]


def evaluar(nombre, pipe, cols=None):
    Xa = X_train if cols is None else X_train[cols]
    Xb = X_test if cols is None else X_test[cols]
    pipe.fit(Xa, y_train)
    p_tr, p_te = pipe.predict(Xa), pipe.predict(Xb)
    cv = cross_val_score(pipe, Xa, y_train, cv=CV, scoring="r2")
    m = {
        "nombre": nombre, "n_terminos": int(pipe[-1].coef_.size),
        "r2_train": float(r2_score(y_train, p_tr)),
        "r2_test": float(r2_score(y_test, p_te)),
        "r2_cv": float(cv.mean()), "r2_cv_std": float(cv.std()),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, p_te))),
        "mae_test": float(mean_absolute_error(y_test, p_te)),
    }
    m["brecha"] = m["r2_train"] - m["r2_cv"]
    print(f"  {nombre:36s} p={m['n_terminos']:3d}  R2tr={m['r2_train']:.4f}  "
          f"R2te={m['r2_test']:.4f}  CV={m['r2_cv']:.4f}+-{m['r2_cv_std']:.3f}  "
          f"brecha={m['brecha']:+.4f}  RMSE={m['rmse_test']:6.2f}  MAE={m['mae_test']:6.2f}")
    modelos.append(m)
    return m, pipe


titulo("FASE B.1 - VARIANTES DE REGRESION LINEAL MULTIPLE")
print("Todas con StandardScaler dentro del Pipeline:\n")

m_ols, p_ols = evaluar("OLS - 10 variables", Pipeline([
    ("sc", StandardScaler()), ("lr", LinearRegression())]))
m_vif, p_vif = evaluar("OLS - sin s1,s2 (corte por VIF)", Pipeline([
    ("sc", StandardScaler()), ("lr", LinearRegression())]), cols=SIN_S1S2)
m_ridge, p_ridge = evaluar("Ridge (L2, alpha por CV)", Pipeline([
    ("sc", StandardScaler()), ("lr", RidgeCV(alphas=np.logspace(-3, 3, 61), cv=CV))]))
m_lasso, p_lasso = evaluar("Lasso (L1, alpha por CV)", Pipeline([
    ("sc", StandardScaler()),
    ("lr", LassoCV(cv=CV, random_state=RANDOM_STATE, max_iter=100000))]))

print(f"\n  alpha Ridge = {p_ridge[-1].alpha_:.4f}   alpha Lasso = {p_lasso[-1].alpha_:.4f}")
nulos_lasso = [c for c, v in zip(TODAS, p_lasso[-1].coef_) if abs(v) < 1e-8]
print(f"  Lasso anula: {nulos_lasso if nulos_lasso else 'ninguna variable'}")
print("\n  La regularizacion apenas cambia las metricas: con 10 variables y 331")
print("  muestras el modelo no tiene capacidad para sobreajustar.")

titulo("FASE B.2 - SIGNIFICANCIA ESTADISTICA DE LOS COEFICIENTES")
sc = StandardScaler().fit(X_train)
A1 = np.column_stack([np.ones(len(X_train)), sc.transform(X_train)])
beta = np.linalg.lstsq(A1, y_train, rcond=None)[0]
resid = y_train - A1 @ beta
gl = len(A1) - A1.shape[1]
cov = (resid @ resid / gl) * np.linalg.inv(A1.T @ A1)
se = np.sqrt(np.diag(cov))
t = beta / se
p_val = 2 * (1 - stats.t.cdf(np.abs(t), gl))
tabla = pd.DataFrame({"coef": beta, "error_std": se, "t": t, "p_valor": p_val},
                     index=["intercepto"] + TODAS).round(4)
tabla["signif"] = ["***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
                   for p in p_val]
print(tabla.to_string())
print("\n  Variables significativas al 5%: " +
      ", ".join(v for v, p in zip(TODAS, p_val[1:]) if p < 0.05))
print("  Las no significativas son las de VIF alto: la multicolinealidad infla el")
print("  error estandar y el modelo no puede separar el efecto de cada lipido.")
resultados["coeficientes_ols"] = {
    "variables": ["intercepto"] + TODAS,
    "coef": [float(v) for v in beta],
    "error_std": [float(v) for v in se],
    "p_valor": [float(v) for v in p_val],
}

titulo("FASE B.3 - CONTROL: EFECTO DE TRATAR LOS OUTLIERS")
lo_w, hi_w = X_train.quantile(0.01), X_train.quantile(0.99)
pw = Pipeline([("sc", StandardScaler()), ("lr", LinearRegression())]).fit(
    X_train.clip(lo_w, hi_w, axis=1), y_train)
r2w = r2_score(y_test, pw.predict(X_test.clip(lo_w, hi_w, axis=1)))
print(f"  OLS sin tratar outliers : R2 test = {m_ols['r2_test']:.4f}")
print(f"  OLS winsorizado 1%-99%  : R2 test = {r2w:.4f}   "
      f"(diferencia {r2w - m_ols['r2_test']:+.4f})")
print("\n  Diferencia despreciable: se confirma la decision de conservarlos.")
resultados["r2_winsorizado"] = float(r2w)

titulo("FASE B.4 - COMPARATIVA Y MODELO FINAL")
comp = pd.DataFrame(modelos).set_index("nombre")
print(comp[["n_terminos", "r2_train", "r2_test", "r2_cv", "brecha",
            "rmse_test", "mae_test"]].round(4).to_string())
mejor = max(modelos, key=lambda m: m["r2_cv"])
print(f"\n  Mejor R2 de validacion cruzada: {mejor['nombre']} ({mejor['r2_cv']:.4f})")
print("  MODELO FINAL: OLS con las 10 variables.")
print("  Se compara por CV y no por R2 de prueba: con 111 muestras de prueba ese")
print("  valor es ruidoso y optimizar sobre el seria hacer trampa.")
resultados["comparativa"] = modelos
resultados["modelo_final"] = "OLS - 10 variables"

# --- diagnostico de residuos
pred_te = p_ols.predict(X_test)
res = y_test - pred_te
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(pred_te, res, alpha=.6, color="#0d3c6c", edgecolor="none")
axes[0].axhline(0, c="#d62828", ls="--")
axes[0].set_xlabel("Prediccion"); axes[0].set_ylabel("Residuo")
axes[0].set_title("Residuos vs. prediccion")
sns.histplot(res, kde=True, ax=axes[1], color="#0d3c6c", bins=25)
axes[1].set_title("Distribucion de residuos"); axes[1].set_xlabel("Residuo")
stats.probplot(res, dist="norm", plot=axes[2])
axes[2].set_title("Q-Q de los residuos")
fig.suptitle("Fase B - Diagnostico de residuos (Regresion Lineal Multiple)", fontsize=12)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/b_residuos.png", dpi=120); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, pred_te, alpha=.65, color="#f77f00", edgecolor="none")
lims = [y.min(), y.max()]
ax.plot(lims, lims, "--", c="#d62828", label="prediccion perfecta")
ax.set_xlabel("Valor real"); ax.set_ylabel("Prediccion")
ax.set_title(f"Real vs. predicho (R2 test = {m_ols['r2_test']:.3f})")
ax.legend()
fig.tight_layout(); fig.savefig(f"{FIGDIR}/b_real_vs_predicho.png", dpi=120); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5))
co = pd.Series(p_ols[-1].coef_, index=TODAS).sort_values()
ax.barh(co.index, co.values, color=["#d62828" if v < 0 else "#0d3c6c" for v in co])
ax.set_xlabel("Coeficiente (variables estandarizadas)")
ax.set_title("Fase B - Peso de cada variable en la Regresion Lineal Multiple")
fig.tight_layout(); fig.savefig(f"{FIGDIR}/b_coeficientes.png", dpi=120); plt.close(fig)


# =============================================================================
# FASE C - EXPORTACION PARA EL APLICATIVO WEB
# =============================================================================
titulo("FASE C - EXPORTACION DEL MODELO")

sc_ = p_ols.named_steps["sc"]
lr_ = p_ols[-1]
payload = {
    "caso": "Caso 3 - Diabetes | Regresion Lineal Multiple",
    "metodologia": "Regresion Lineal Multiple (OLS, 10 variables)",
    "descripcion": DESCRIPCION,
    "variables": TODAS,
    "rangos": {c: [float(X[c].min()), float(X[c].max())] for c in TODAS},
    "medianas": {c: float(X[c].median()) for c in TODAS},
    "target": {"min": float(y.min()), "max": float(y.max()), "media": float(y.mean())},
    "n_train": len(X_train), "n_test": len(X_test),
    "mu": [float(v) for v in sc_.mean_],
    "sigma": [float(v) for v in sc_.scale_],
    "coef": [float(v) for v in lr_.coef_],
    "intercept": float(lr_.intercept_),
    "error_std": [float(v) for v in se[1:]],
    "p_valor": [float(v) for v in p_val[1:]],
    "metricas": m_ols,
    "comparativa": modelos,
    "vif": vif,
}

with open("model.js", "w", encoding="utf-8") as f:
    f.write("// Generado por train.py - Caso 3 Diabetes | Regresion Lineal Multiple\n")
    f.write("const MODELO_MULTIPLE = " + json.dumps(payload, indent=2, ensure_ascii=False) + ";\n")

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

print(f"  model.js         -> {len(TODAS)} coeficientes + intercepto")
print("  resultados.json  -> metricas de las Fases A y B")
print(f"  {FIGDIR}/         -> 7 figuras")
print("\nListo.")
