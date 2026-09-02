"""
CASO 1 - CALIFORNIA HOUSING  |  FASES A y B COMPLETAS
Universidad Andina del Cusco - Inteligencia Artificial

Completa lo que faltaba del caso:
  Fase A - deteccion de outliers y analisis de multicolinealidad (correlacion + VIF).
  Fase B - Regresion Lineal Multiple ademas de la Polinomica, ambas bajo el
           MISMO protocolo (particion 80/20, random_state=42, identico
           preprocesamiento) para que la comparacion sea legitima.

No sustituye a entrenar.py: ese sigue siendo el que genera el modelo desplegado.
Este script produce las cifras y figuras que documenta el informe.

Ejecutar:  python analisis_fases_a_b.py
Genera:    figuras_analisis/*.png, resultados_fases_a_b.json
"""

import json
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

RANDOM_STATE = 42
CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
FIGDIR = "figuras_analisis"
os.makedirs(FIGDIR, exist_ok=True)
sns.set_theme(style="whitegrid")

NUMERICAS = ["longitude", "latitude", "housing_median_age", "total_rooms",
             "total_bedrooms", "population", "households", "median_income"]
CATEGORICA = "ocean_proximity"
OBJETIVO = "median_house_value"

res = {}


def titulo(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# =============================================================================
# FASE A
# =============================================================================
titulo("FASE A.1 - CARGA, TIPOS DE DATOS Y NULOS")
df = pd.read_csv("housing.csv")
print(f"Registros: {len(df)}   Columnas: {len(df.columns)}")
print(pd.DataFrame({"dtype": df.dtypes.astype(str), "nulos": df.isna().sum()}).to_string())

nulos = int(df["total_bedrooms"].isna().sum())
print(f"\nNulos en total_bedrooms: {nulos} ({100*nulos/len(df):.2f} %)")
print("Se imputan con la MEDIANA dentro del Pipeline: asi la mediana se calcula")
print("solo con los datos de entrenamiento y no hay fuga de informacion.")
print(f"Duplicados: {int(df.duplicated().sum())}")
res["n_registros"] = len(df)
res["nulos_total_bedrooms"] = nulos
res["duplicados"] = int(df.duplicated().sum())

titulo("FASE A.2 - DETECCION DE OUTLIERS (metodo IQR, k = 1.5)")
outliers = {}
print(f"{'Variable':<22}{'Outliers':>10}{'%':>8}   Limites [inferior, superior]")
for c in NUMERICAS + [OBJETIVO]:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n = int(((df[c] < lo) | (df[c] > hi)).sum())
    outliers[c] = n
    print(f"{c:<22}{n:>10}{100*n/len(df):>7.2f}%   [{lo:,.2f}, {hi:,.2f}]")
res["outliers"] = outliers

# El target esta censurado: todos los distritos por encima del tope se
# registraron con el mismo valor. No es un outlier estadistico sino un
# artefacto administrativo del censo.
censurados = int((df[OBJETIVO] >= 500001).sum())
print(f"\nRegistros con {OBJETIVO} = 500 001 (tope del censo): {censurados} "
      f"({100*censurados/len(df):.2f} %)")
print("Decision: se CONSERVAN los outliers (son distritos grandes o ricos reales)")
print("y se documenta la censura del target como limitacion del dataset.")
res["censurados"] = censurados

fig, axes = plt.subplots(2, 4, figsize=(17, 7))
for ax, c in zip(axes.ravel(), NUMERICAS):
    sns.boxplot(y=df[c], ax=ax, color="#0d3c6c", width=.5)
    ax.set_title(f"{c}  ({outliers[c]} outliers)", fontsize=10)
    ax.set_ylabel("")
fig.suptitle("Caso 1 · Fase A — Diagramas de caja (metodo IQR)", fontsize=13)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/a_outliers.png", dpi=110); plt.close(fig)

titulo("FASE A.3 - ANALISIS DE MULTICOLINEALIDAD")
corr = df[NUMERICAS].corr()
print("Pares con |r| > 0.5:")
pares = []
for i in range(len(corr)):
    for j in range(i + 1, len(corr)):
        r = corr.iloc[i, j]
        if abs(r) > 0.5:
            pares.append([corr.index[i], corr.columns[j], round(float(r), 4)])
            print(f"  {corr.index[i]:<20} - {corr.columns[j]:<20} r = {r:+.4f}")
res["correlaciones_altas"] = pares

# VIF_j = 1 / (1 - R2_j): se regresa cada variable contra las demas.
X_vif = df[NUMERICAS].fillna(df[NUMERICAS].median())
Z = StandardScaler().fit_transform(X_vif)
vif = {}
for j, nom in enumerate(NUMERICAS):
    otras = [k for k in range(Z.shape[1]) if k != j]
    r2 = LinearRegression().fit(Z[:, otras], Z[:, j]).score(Z[:, otras], Z[:, j])
    vif[nom] = round(float(1 / (1 - r2)), 2)
print("\nFactor de Inflacion de la Varianza (VIF):")
for k, v in sorted(vif.items(), key=lambda p: -p[1]):
    marca = "  <-- SEVERA" if v > 10 else ("  <-- moderada" if v > 5 else "")
    print(f"  {k:<22}{v:>9.2f}{marca}")
print("\n  total_rooms, total_bedrooms, population y households miden todos el")
print("  TAMANO del distrito, de ahi su VIF elevado. Es multicolinealidad")
print("  esperable y no impide predecir, pero desaconseja interpretar esos")
print("  coeficientes por separado.")
res["vif"] = vif

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True,
            linewidths=.5, ax=ax, cbar_kws={"shrink": .8})
ax.set_title("Caso 1 · Fase A — Matriz de correlacion de Pearson", fontsize=12)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/a_correlacion.png", dpi=115); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5))
s = pd.Series(vif).sort_values(ascending=False)
ax.bar(s.index, s.values,
       color=["#d62828" if v > 10 else "#f77f00" if v > 5 else "#0d3c6c" for v in s])
ax.axhline(10, ls="--", c="#d62828", label="VIF = 10 (severa)")
ax.axhline(5, ls=":", c="#f77f00", label="VIF = 5 (moderada)")
ax.set_ylabel("VIF"); ax.set_title("Caso 1 · Fase A — Factor de Inflacion de la Varianza")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right"); ax.legend()
fig.tight_layout(); fig.savefig(f"{FIGDIR}/a_vif.png", dpi=115); plt.close(fig)


# =============================================================================
# FASE B - AMBAS METODOLOGIAS BAJO EL MISMO PROTOCOLO
# =============================================================================
titulo("FASE B - PARTICION Y PREPROCESAMIENTO COMUN")
X = df[NUMERICAS + [CATEGORICA]]
y = df[OBJETIVO]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
print(f"Entrenamiento: {len(X_tr)}   Prueba: {len(X_te)}   (80 / 20)")
print("\nAmbos modelos comparten exactamente el mismo preprocesamiento:")
print("  numericas  -> SimpleImputer(mediana) -> StandardScaler [-> PolynomialFeatures]")
print("  categorica -> OneHotEncoder")
print("La UNICA diferencia entre ellos es la expansion polinomica.")
res["n_train"], res["n_test"] = len(X_tr), len(X_te)


def construir(grado):
    pasos = [("imputer", SimpleImputer(strategy="median")),
             ("scaler", StandardScaler())]
    if grado > 1:
        pasos.append(("poly", PolynomialFeatures(degree=grado, include_bias=False)))
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline(pasos), NUMERICAS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), [CATEGORICA]),
        ])),
        ("modelo", LinearRegression()),
    ])


def evaluar(nombre, grado):
    pipe = construir(grado)
    pipe.fit(X_tr, y_tr)
    p_tr, p_te = pipe.predict(X_tr), pipe.predict(X_te)
    cv = cross_val_score(pipe, X_tr, y_tr, cv=CV, scoring="r2")
    m = {
        "nombre": nombre,
        "n_terminos": int(pipe.named_steps["modelo"].coef_.size),
        "r2_train": round(float(r2_score(y_tr, p_tr)), 4),
        "r2_test": round(float(r2_score(y_te, p_te)), 4),
        "r2_cv": round(float(cv.mean()), 4),
        "r2_cv_std": round(float(cv.std()), 4),
        "rmse_test": round(float(np.sqrt(mean_squared_error(y_te, p_te))), 2),
        "mae_test": round(float(mean_absolute_error(y_te, p_te)), 2),
    }
    m["brecha"] = round(m["r2_train"] - m["r2_cv"], 4)
    print(f"  {nombre:<34} p={m['n_terminos']:>3}  R2tr={m['r2_train']:.4f}  "
          f"R2te={m['r2_test']:.4f}  CV={m['r2_cv']:.4f}+-{m['r2_cv_std']:.3f}  "
          f"RMSE=${m['rmse_test']:>10,.2f}  MAE=${m['mae_test']:>10,.2f}")
    return m, pipe


titulo("FASE B.1 - REGRESION LINEAL MULTIPLE (modelo base)")
m_lin, pipe_lin = evaluar("Lineal multiple (OLS)", 1)

titulo("FASE B.2 - REGRESION POLINOMICA GRADO 2")
m_poly, pipe_poly = evaluar("Polinomica grado 2 (OLS)", 2)

titulo("FASE B.3 - COMPARATIVA")
tabla = pd.DataFrame([m_lin, m_poly]).set_index("nombre")
print(tabla[["n_terminos", "r2_train", "r2_test", "r2_cv", "brecha",
             "rmse_test", "mae_test"]].to_string())

d_r2 = m_poly["r2_test"] - m_lin["r2_test"]
d_mae = m_poly["mae_test"] - m_lin["mae_test"]
print(f"\n  Ganancia de la expansion polinomica sobre la lineal multiple:")
print(f"    R2 de prueba : {d_r2:+.4f}  ({d_r2*100:+.2f} puntos porcentuales)")
print(f"    MAE          : {d_mae:+,.2f} US$  "
      f"({'mejora' if d_mae < 0 else 'empeora'})")
print(f"    Terminos     : {m_lin['n_terminos']} -> {m_poly['n_terminos']}")
print(f"\n  Brecha entrenamiento - CV:  lineal {m_lin['brecha']:.4f}  |  "
      f"polinomica {m_poly['brecha']:.4f}")
res["fase_b"] = {"lineal": m_lin, "polinomica": m_poly,
                 "delta_r2_test": round(d_r2, 4), "delta_mae": round(d_mae, 2)}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
et = ["Lineal múltiple\n(%d términos)" % m_lin["n_terminos"],
      "Polinómica grado 2\n(%d términos)" % m_poly["n_terminos"]]
x = np.arange(2); an = 0.35
axes[0].bar(x - an/2, [m_lin["r2_train"], m_poly["r2_train"]], an,
            label="Entrenamiento", color="#0d3c6c")
axes[0].bar(x + an/2, [m_lin["r2_test"], m_poly["r2_test"]], an,
            label="Prueba", color="#f77f00")
axes[0].set_xticks(x); axes[0].set_xticklabels(et); axes[0].set_ylabel("R²")
axes[0].set_title("Caso 1 · Fase B — Ajuste por modelo"); axes[0].legend()
for i, v in enumerate([m_lin["r2_train"], m_poly["r2_train"]]):
    axes[0].annotate(f"{v:.4f}", (i - an/2, v), ha="center", va="bottom", fontsize=9)
for i, v in enumerate([m_lin["r2_test"], m_poly["r2_test"]]):
    axes[0].annotate(f"{v:.4f}", (i + an/2, v), ha="center", va="bottom", fontsize=9)

axes[1].bar(et, [m_lin["mae_test"], m_poly["mae_test"]], color=["#0d3c6c", "#1d9e6f"])
axes[1].set_ylabel("MAE de prueba (US$)")
axes[1].set_title("Caso 1 · Fase B — Error absoluto medio")
for i, v in enumerate([m_lin["mae_test"], m_poly["mae_test"]]):
    axes[1].annotate(f"${v:,.0f}", (i, v), ha="center", va="bottom", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/b_comparativa.png", dpi=115); plt.close(fig)

with open("resultados_fases_a_b.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)

titulo("ARCHIVOS GENERADOS")
print("  resultados_fases_a_b.json")
print(f"  {FIGDIR}/a_outliers.png · a_correlacion.png · a_vif.png · b_comparativa.png")
