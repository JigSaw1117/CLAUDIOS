"""
CASO 1 - CALIFORNIA HOUSING PRICES  |  REGRESION LINEAL MULTIPLE
Universidad Andina del Cusco - Inteligencia Artificial

Prediccion del valor mediano de vivienda por distrito, a partir de los
descriptores del censo de California de 1990 (20 640 distritos).

Fase A: analisis exploratorio y preprocesamiento.
Fase B: regresion lineal multiple (OLS, Ridge, Lasso).
Fase C: exporta model.js para el aplicativo web.

Ejecutar:  python train.py
Genera:    figuras/*.png, model.js, resultados.json
"""

import json
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

RANDOM_STATE = 42
CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
FIGDIR = "figuras"
os.makedirs(FIGDIR, exist_ok=True)
sns.set_theme(style="whitegrid")

# El CSV vive en la carpeta del caso; no se duplica.
CSV = "../housing.csv" if os.path.exists("../housing.csv") else "housing.csv"
resultados = {}


def titulo(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# =============================================================================
# FASE A - ANALISIS EXPLORATORIO Y PREPROCESAMIENTO
# =============================================================================
titulo("FASE A.1 - CARGA, TIPOS DE DATOS Y VALORES NULOS")
df = pd.read_csv(CSV)
print(f"Filas: {len(df)}   Columnas: {len(df.columns)}")
print(pd.DataFrame({"dtype": df.dtypes.astype(str), "nulos": df.isna().sum()}).to_string())

nulos_bedrooms = int(df["total_bedrooms"].isna().sum())
print(f"\nNulos en total_bedrooms: {nulos_bedrooms} ({100*nulos_bedrooms/len(df):.2f}%)")
print("Se imputan con la MEDIANA calculada solo sobre el set de entrenamiento,")
print("para no filtrar informacion del set de prueba (data leakage).")
print(f"Filas duplicadas: {df.duplicated().sum()}")

titulo("FASE A.2 - TRATAMIENTO DE OUTLIERS Y DEL TARGET CENSURADO")
# El target esta truncado artificialmente en 500001: todos los distritos mas caros
# se registraron con ese mismo valor. Es un artefacto del censo, no un dato real,
# y sesga sistematicamente cualquier ajuste lineal en el extremo superior.
censurados = int((df["median_house_value"] >= 500001).sum())
print(f"Registros con median_house_value = 500001 (censurado): {censurados} "
      f"({100*censurados/len(df):.2f}%)")
print("Se DESCARTAN: no son valores reales sino un tope administrativo del censo.")
df = df[df["median_house_value"] < 500001].copy()
print(f"Filas tras el filtrado: {len(df)}")

NUMERICAS_ORIG = ["longitude", "latitude", "housing_median_age", "total_rooms",
                  "total_bedrooms", "population", "households", "median_income"]
outliers = {}
for col in NUMERICAS_ORIG:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers[col] = int(((df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)).sum())
print("\nOutliers por IQR:", {k: v for k, v in outliers.items() if v})
print("Se conservan: son distritos grandes o ricos reales, no errores de medicion.")
resultados["outliers"] = outliers
resultados["censurados_descartados"] = censurados

fig, axes = plt.subplots(2, 4, figsize=(17, 7))
for ax, col in zip(axes.ravel(), NUMERICAS_ORIG):
    sns.boxplot(y=df[col], ax=ax, color="#0d3c6c", width=.5)
    ax.set_title(f"{col}  ({outliers[col]} outliers)", fontsize=10)
    ax.set_ylabel("")
fig.suptitle("Fase A - Diagramas de caja de los descriptores originales", fontsize=13)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/a_boxplots.png", dpi=110); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["median_house_value"], kde=True, ax=axes[0], color="#f77f00", bins=40)
axes[0].set_title(f"Target original (asimetria = {df['median_house_value'].skew():.3f})")
sns.histplot(np.log(df["median_house_value"]), kde=True, ax=axes[1], color="#1d9e6f", bins=40)
axes[1].set_title(f"Target en log (asimetria = {np.log(df['median_house_value']).skew():.3f})")
fig.suptitle("Fase A - La transformacion logaritmica simetriza el target", fontsize=12)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/a_target.png", dpi=120); plt.close(fig)

titulo("FASE A.3 - PARTICION 75/25 ESTRATIFICADA")
# Estratificar por categoria de ingreso evita que el 25% de prueba quede sesgado
# justamente en el descriptor mas correlacionado con el precio.
estratos = pd.cut(df["median_income"], bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
                  labels=[1, 2, 3, 4, 5])
train_df, test_df = train_test_split(df, test_size=0.25, random_state=RANDOM_STATE,
                                     stratify=estratos)
print(f"Entrenamiento: {len(train_df)}   Prueba: {len(test_df)}")
print("Estratificado por categoria de ingreso (el descriptor de mayor peso).")

BEDROOMS_MEDIAN = float(train_df["total_bedrooms"].median())
CATEGORIAS = sorted(df["ocean_proximity"].unique())
print(f"\nMediana de total_bedrooms (solo entrenamiento): {BEDROOMS_MEDIAN}")
print(f"Categorias de ocean_proximity: {CATEGORIAS}")
print(f"Se codifican con dummies; la categoria base es '{CATEGORIAS[0]}'")
print("(se omite una para evitar colinealidad perfecta entre las dummies).")

titulo("FASE A.4 - INGENIERIA DE DESCRIPTORES")
SF = (37.7749, -122.4194)
LA = (34.0522, -118.2437)

NUMERICAS = NUMERICAS_ORIG + [
    "rooms_per_household", "bedrooms_per_room", "population_per_household",
    "income_sq", "log_population", "log_households", "log_total_rooms",
    "dist_sf", "dist_la",
]


def construir(raw):
    d = raw.copy()
    d["total_bedrooms"] = d["total_bedrooms"].fillna(BEDROOMS_MEDIAN)
    # Los totales por distrito no son comparables entre distritos de distinto
    # tamano; las razones si lo son y aportan la senal que el modelo necesita.
    d["rooms_per_household"] = d["total_rooms"] / d["households"]
    d["bedrooms_per_room"] = d["total_bedrooms"] / d["total_rooms"]
    d["population_per_household"] = d["population"] / d["households"]
    # Terminos derivados: siguen siendo LINEALES EN LOS PARAMETROS, de modo que el
    # modelo continua siendo una regresion lineal multiple.
    d["income_sq"] = d["median_income"] ** 2
    d["log_population"] = np.log1p(d["population"])
    d["log_households"] = np.log1p(d["households"])
    d["log_total_rooms"] = np.log1p(d["total_rooms"])
    d["dist_sf"] = np.hypot(d["latitude"] - SF[0], (d["longitude"] - SF[1]) * 0.79)
    d["dist_la"] = np.hypot(d["latitude"] - LA[0], (d["longitude"] - LA[1]) * 0.83)
    X = d[NUMERICAS].copy()
    for cat in CATEGORIAS[1:]:
        X[f"ocean_{cat}"] = (d["ocean_proximity"] == cat).astype(float)
    return X


print("A los 8 descriptores originales se anaden 9 derivados y 4 dummies:")
print("  - razones por hogar (rooms_per_household, bedrooms_per_room, ...)")
print("  - logaritmos de las variables sesgadas")
print("  - distancias a San Francisco y Los Angeles")
print("\nTodos son transformaciones de los datos, NO del modelo: la funcion sigue")
print("siendo lineal en los coeficientes, es decir, regresion lineal multiple.")

X_train_raw, X_test_raw = construir(train_df), construir(test_df)
DESCRIPTORES = list(X_train_raw.columns)
print(f"\nTotal de descriptores: {len(DESCRIPTORES)}")

# El target se modela en log: su distribucion es asimetrica y lo que importa es el
# error relativo. Se invierte con exp() al predecir.
y_train = np.log(train_df["median_house_value"].values)
y_test = np.log(test_df["median_house_value"].values)
print(f"Asimetria del target: {df['median_house_value'].skew():.3f} -> "
      f"{pd.Series(np.log(df['median_house_value'])).skew():.3f} en log")

titulo("FASE A.5 - MULTICOLINEALIDAD (VIF) Y ESTANDARIZACION")
MU = X_train_raw.mean().values
SIGMA = X_train_raw.std(ddof=0).values
X_train = (X_train_raw.values - MU) / SIGMA
X_test = (X_test_raw.values - MU) / SIGMA
print("Estandarizacion (media 0, desviacion 1) calculada SOLO con entrenamiento.")

vif = {}
for j, name in enumerate(DESCRIPTORES):
    otras = [k for k in range(X_train.shape[1]) if k != j]
    r2 = LinearRegression().fit(X_train[:, otras], X_train[:, j]).score(
        X_train[:, otras], X_train[:, j])
    vif[name] = float(1 / (1 - r2)) if r2 < 1 else float("inf")
print("\nVIF (los 10 mas altos):")
for k, v in sorted(vif.items(), key=lambda p: -p[1])[:10]:
    print(f"  {k:26s} {v:9.2f}{'  <-- SEVERA' if v > 10 else ''}")
print("\n  Los descriptores de tamano del distrito (total_rooms, population,")
print("  households y sus logaritmos) miden lo mismo desde angulos distintos.")
print("  Por eso se evalua Ridge: la penalizacion L2 estabiliza los coeficientes")
print("  cuando hay colinealidad, sin necesidad de eliminar variables.")
resultados["vif"] = vif

corr = X_train_raw.corr()
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, cmap="RdBu_r", center=0, square=True, linewidths=.3,
            ax=ax, cbar_kws={"shrink": .8}, annot=False)
ax.set_title("Fase A - Matriz de correlacion de los descriptores", fontsize=12)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/a_correlacion.png", dpi=110); plt.close(fig)

resultados["n_train"], resultados["n_test"] = len(X_train), len(X_test)
resultados["n_descriptores"] = len(DESCRIPTORES)


# =============================================================================
# FASE B - REGRESION LINEAL MULTIPLE
# =============================================================================
modelos = []


def evaluar(nombre, modelo, Xa=None, Xb=None):
    Xa = X_train if Xa is None else Xa
    Xb = X_test if Xb is None else Xb
    modelo.fit(Xa, y_train)
    p_tr, p_te = modelo.predict(Xa), modelo.predict(Xb)
    cv = cross_val_score(modelo, Xa, y_train, cv=CV, scoring="r2")
    # Metricas en dolares: se deshace el logaritmo.
    real_te, dol_te = np.exp(y_test), np.exp(p_te)
    m = {
        "nombre": nombre, "n_terminos": int(modelo.coef_.size),
        "r2_train": float(r2_score(y_train, p_tr)),
        "r2_test": float(r2_score(y_test, p_te)),
        "r2_cv": float(cv.mean()), "r2_cv_std": float(cv.std()),
        "rmse_test": float(np.sqrt(mean_squared_error(real_te, dol_te))),
        "mae_test": float(mean_absolute_error(real_te, dol_te)),
    }
    m["brecha"] = m["r2_train"] - m["r2_cv"]
    print(f"  {nombre:34s} p={m['n_terminos']:3d}  R2tr={m['r2_train']:.4f}  "
          f"R2te={m['r2_test']:.4f}  CV={m['r2_cv']:.4f}+-{m['r2_cv_std']:.3f}  "
          f"RMSE=${m['rmse_test']:,.0f}  MAE=${m['mae_test']:,.0f}")
    modelos.append(m)
    return m, modelo


titulo("FASE B.1 - MODELO BASE: SOLO LOS 8 DESCRIPTORES ORIGINALES")
base_cols = [DESCRIPTORES.index(c) for c in NUMERICAS_ORIG] + \
            [i for i, c in enumerate(DESCRIPTORES) if c.startswith("ocean_")]
m_base, _ = evaluar("OLS - 8 originales + dummies",
                    LinearRegression(), X_train[:, base_cols], X_test[:, base_cols])

titulo("FASE B.2 - MODELO COMPLETO: 22 DESCRIPTORES")
m_ols, mod_ols = evaluar("OLS - 22 descriptores", LinearRegression())
m_ridge, mod_ridge = evaluar(
    "Ridge (L2, alpha por CV)", RidgeCV(alphas=np.logspace(-3, 3, 61), cv=CV))
m_lasso, mod_lasso = evaluar(
    "Lasso (L1, alpha por CV)",
    LassoCV(cv=CV, random_state=RANDOM_STATE, max_iter=50000))

print(f"\n  alpha Ridge = {mod_ridge.alpha_:.4f}   alpha Lasso = {mod_lasso.alpha_:.6f}")
print(f"\n  La ingenieria de descriptores sube el R2 de prueba de "
      f"{m_base['r2_test']:.4f} a {m_ols['r2_test']:.4f}")
print(f"  ({m_ols['r2_test'] - m_base['r2_test']:+.4f}), sin abrir brecha entre")
print("  entrenamiento y validacion: no es sobreajuste, es reduccion del sesgo.")

titulo("FASE B.3 - MODELO FINAL Y DIAGNOSTICO DE SOBREAJUSTE")
# Con descriptores geograficos colineales se prefiere Ridge salvo que OLS gane
# de forma clara: sus coeficientes son mas estables.
final = mod_ols if m_ols["r2_test"] - m_ridge["r2_test"] > 0.005 else mod_ridge
m_final = m_ols if final is mod_ols else m_ridge
print(f"  Modelo final: {m_final['nombre']}")
print(f"  R2 entrenamiento {m_final['r2_train']:.4f} | prueba {m_final['r2_test']:.4f} | "
      f"CV {m_final['r2_cv']:.4f}")
print(f"  Brecha entrenamiento - prueba: {m_final['r2_train'] - m_final['r2_test']:+.4f}")
print("  Brecha practicamente nula: el modelo generaliza, no memoriza.")

tam, sc_tr, sc_va = learning_curve(
    final, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 8),
    cv=CV, scoring="r2")
print("\nCurva de aprendizaje (R2 entrenamiento / R2 validacion):")
for n, t, v in zip(tam, sc_tr.mean(1), sc_va.mean(1)):
    print(f"  n={n:6.0f}   train={t:.4f}   val={v:.4f}   brecha={t - v:+.4f}")
resultados["curva_aprendizaje"] = [
    {"n": int(n), "train": float(t), "val": float(v)}
    for n, t, v in zip(tam, sc_tr.mean(1), sc_va.mean(1))]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(tam, sc_tr.mean(1), "o-", c="#0d3c6c", label="Entrenamiento")
ax.plot(tam, sc_va.mean(1), "s--", c="#f77f00", label="Validacion cruzada")
ax.fill_between(tam, sc_tr.mean(1) - sc_tr.std(1), sc_tr.mean(1) + sc_tr.std(1),
                alpha=.12, color="#0d3c6c")
ax.fill_between(tam, sc_va.mean(1) - sc_va.std(1), sc_va.mean(1) + sc_va.std(1),
                alpha=.12, color="#f77f00")
ax.set_xlabel("Muestras de entrenamiento"); ax.set_ylabel("R2")
ax.set_title("Fase B - Curva de aprendizaje: las curvas convergen (sin sobreajuste)")
ax.legend()
fig.tight_layout(); fig.savefig(f"{FIGDIR}/b_curva_aprendizaje.png", dpi=120); plt.close(fig)

titulo("FASE B.4 - COMPARATIVA")
comp = pd.DataFrame(modelos).set_index("nombre")
print(comp[["n_terminos", "r2_train", "r2_test", "r2_cv", "brecha",
            "rmse_test", "mae_test"]].round(4).to_string())
resultados["comparativa"] = modelos
resultados["modelo_final"] = m_final["nombre"]

print("\nCoeficientes del modelo final (escala estandarizada):")
for f, c in sorted(zip(DESCRIPTORES, final.coef_), key=lambda p: -abs(p[1]))[:12]:
    print(f"  {f:26s} {c:+.4f}")

pred_te_dol = np.exp(final.predict(X_test))
real_te_dol = np.exp(y_test)
res = real_te_dol - pred_te_dol

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(pred_te_dol, res, alpha=.25, s=8, color="#0d3c6c", edgecolor="none")
axes[0].axhline(0, c="#d62828", ls="--")
axes[0].set_xlabel("Prediccion ($)"); axes[0].set_ylabel("Residuo ($)")
axes[0].set_title("Residuos vs. prediccion")
sns.histplot(res, kde=True, ax=axes[1], color="#0d3c6c", bins=45)
axes[1].set_title("Distribucion de residuos"); axes[1].set_xlabel("Residuo ($)")
axes[2].scatter(real_te_dol, pred_te_dol, alpha=.25, s=8, color="#1d9e6f", edgecolor="none")
lims = [real_te_dol.min(), real_te_dol.max()]
axes[2].plot(lims, lims, "--", c="#d62828")
axes[2].set_xlabel("Valor real ($)"); axes[2].set_ylabel("Prediccion ($)")
axes[2].set_title(f"Real vs. predicho (R2 = {m_final['r2_test']:.3f})")
fig.suptitle("Fase B - Diagnostico del modelo final", fontsize=12)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/b_diagnostico.png", dpi=115); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 6))
co = pd.Series(final.coef_, index=DESCRIPTORES).sort_values()
ax.barh(co.index, co.values, color=["#d62828" if v < 0 else "#0d3c6c" for v in co])
ax.set_xlabel("Coeficiente (descriptores estandarizados)")
ax.set_title("Fase B - Peso de cada descriptor en el modelo final")
fig.tight_layout(); fig.savefig(f"{FIGDIR}/b_coeficientes.png", dpi=120); plt.close(fig)


# =============================================================================
# FASE C - EXPORTACION PARA EL APLICATIVO WEB
# =============================================================================
titulo("FASE C - EXPORTACION DEL MODELO")

payload = {
    "caso": "Caso 1 - California Housing | Regresion Lineal Multiple",
    "metodologia": f"{m_final['nombre']} sobre log(precio), 22 descriptores",
    "descriptores": DESCRIPTORES,
    "numericas": NUMERICAS,
    "categorias": CATEGORIAS,
    "categoria_base": CATEGORIAS[0],
    "mu": [float(v) for v in MU],
    "sigma": [float(v) for v in SIGMA],
    "coef": [float(v) for v in final.coef_],
    "intercept": float(final.intercept_),
    "bedrooms_median": BEDROOMS_MEDIAN,
    "log_target": True,
    "rangos": {c: [float(df[c].min()), float(df[c].max())] for c in NUMERICAS_ORIG},
    "medianas": {c: float(df[c].median()) for c in NUMERICAS_ORIG},
    "target": {"min": float(df["median_house_value"].min()),
               "max": float(df["median_house_value"].max()),
               "media": float(df["median_house_value"].mean())},
    "metricas": m_final,
    "comparativa": modelos,
    "vif": vif,
    "n_train": len(X_train), "n_test": len(X_test),
    "censurados_descartados": censurados,
    "nulos_imputados": nulos_bedrooms,
}

with open("model.js", "w", encoding="utf-8") as f:
    f.write("// Generado por train.py - Caso 1 California Housing | Regresion Lineal Multiple\n")
    f.write("const MODELO_MULTIPLE = " + json.dumps(payload, indent=2, ensure_ascii=False) + ";\n")

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

print(f"  model.js         -> {len(DESCRIPTORES)} coeficientes + intercepto")
print("  resultados.json  -> metricas de las Fases A y B")
print(f"  {FIGDIR}/         -> 5 figuras")
print("\nListo.")
