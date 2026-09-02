"""
CASO 2 - WINE QUALITY  |  FASES A y B COMPLETAS
Universidad Andina del Cusco - Inteligencia Artificial

Completa lo que faltaba del caso:
  Fase A - deteccion de outliers y analisis de multicolinealidad (correlacion + VIF).
  Fase B - Regresion Lineal Multiple y Polinomica grado 2 bajo el mismo
           protocolo (80/20, random_state=42), reproduciendo las cifras que
           consumen wine_model_linear.json y wine_model_poly.json.

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

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

RANDOM_STATE = 42
CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
FIGDIR = "figuras_analisis"
os.makedirs(FIGDIR, exist_ok=True)
sns.set_theme(style="whitegrid")

OBJETIVO = "quality"
res = {}


def titulo(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# =============================================================================
# FASE A
# =============================================================================
titulo("FASE A.1 - CARGA, LIMPIEZA Y NULOS")
df = pd.read_csv("WineQT.csv")
print(f"Registros: {len(df)}   Columnas: {len(df.columns)}")

# 'Id' es un identificador de fila: no aporta informacion predictiva y su
# inclusion solo anadiria ruido al modelo.
if "Id" in df.columns:
    df = df.drop(columns=["Id"])
    print("Se elimina la columna 'Id' (identificador sin valor predictivo).")

PREDICTORES = [c for c in df.columns if c != OBJETIVO]
print(f"Predictores: {len(PREDICTORES)}   Objetivo: {OBJETIVO}")
print(pd.DataFrame({"dtype": df.dtypes.astype(str), "nulos": df.isna().sum()}).to_string())
print(f"\nNulos totales: {int(df.isna().sum().sum())}   Duplicados: {int(df.duplicated().sum())}")
print(f"Rango del objetivo: {df[OBJETIVO].min()} - {df[OBJETIVO].max()}  "
      f"(escala discreta valorada por catadores)")
print("Distribucion de la calidad:")
print(df[OBJETIVO].value_counts().sort_index().to_string())
res["n_registros"] = len(df)
res["nulos"] = int(df.isna().sum().sum())
res["duplicados"] = int(df.duplicated().sum())
res["distribucion_objetivo"] = {int(k): int(v) for k, v in
                                df[OBJETIVO].value_counts().sort_index().items()}

titulo("FASE A.2 - DETECCION DE OUTLIERS (metodo IQR, k = 1.5)")
outliers = {}
print(f"{'Variable':<24}{'Outliers':>10}{'%':>8}   Limites [inferior, superior]")
for c in PREDICTORES:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n = int(((df[c] < lo) | (df[c] > hi)).sum())
    outliers[c] = n
    print(f"{c:<24}{n:>10}{100*n/len(df):>7.2f}%   [{lo:,.3f}, {hi:,.3f}]")
total_out = sum(outliers.values())
print(f"\nTotal de valores atipicos: {total_out} sobre {len(df)*len(PREDICTORES)} celdas "
      f"({100*total_out/(len(df)*len(PREDICTORES)):.2f} %)")
print("Decision: se CONSERVAN. Son composiciones quimicas reales (un vino puede")
print("tener acidez volatil alta y seguir siendo un vino valido); eliminarlos")
print("sesgaria el modelo hacia el vino promedio y reduciria una muestra ya pequena.")
res["outliers"] = outliers

fig, axes = plt.subplots(3, 4, figsize=(17, 10))
for ax, c in zip(axes.ravel(), PREDICTORES):
    sns.boxplot(y=df[c], ax=ax, color="#8b1a35", width=.5)
    ax.set_title(f"{c}\n({outliers[c]} outliers)", fontsize=9)
    ax.set_ylabel("")
for ax in axes.ravel()[len(PREDICTORES):]:
    ax.axis("off")
fig.suptitle("Caso 2 · Fase A — Diagramas de caja (metodo IQR)", fontsize=13)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/a_outliers.png", dpi=110); plt.close(fig)

titulo("FASE A.3 - ANALISIS DE MULTICOLINEALIDAD")
corr = df[PREDICTORES].corr()
print("Pares con |r| > 0.5:")
pares = []
for i in range(len(corr)):
    for j in range(i + 1, len(corr)):
        r = corr.iloc[i, j]
        if abs(r) > 0.5:
            pares.append([corr.index[i], corr.columns[j], round(float(r), 4)])
            print(f"  {corr.index[i]:<22} - {corr.columns[j]:<22} r = {r:+.4f}")
if not pares:
    print("  (ninguno)")
res["correlaciones_altas"] = pares

print("\nCorrelacion de cada predictor con la calidad:")
corr_obj = df[PREDICTORES + [OBJETIVO]].corr()[OBJETIVO].drop(OBJETIVO).sort_values()
for k, v in corr_obj.items():
    print(f"  {k:<24}{v:+.4f}")
res["correlacion_con_objetivo"] = {k: round(float(v), 4) for k, v in corr_obj.items()}

Z = StandardScaler().fit_transform(df[PREDICTORES])
vif = {}
for j, nom in enumerate(PREDICTORES):
    otras = [k for k in range(Z.shape[1]) if k != j]
    r2 = LinearRegression().fit(Z[:, otras], Z[:, j]).score(Z[:, otras], Z[:, j])
    vif[nom] = round(float(1 / (1 - r2)), 2)
print("\nFactor de Inflacion de la Varianza (VIF):")
for k, v in sorted(vif.items(), key=lambda p: -p[1]):
    marca = "  <-- SEVERA" if v > 10 else ("  <-- moderada" if v > 5 else "")
    print(f"  {k:<24}{v:>9.2f}{marca}")
res["vif"] = vif

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True,
            linewidths=.5, ax=ax, cbar_kws={"shrink": .8}, annot_kws={"size": 7})
ax.set_title("Caso 2 · Fase A — Matriz de correlacion de Pearson", fontsize=12)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/a_correlacion.png", dpi=115); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.5))
s = pd.Series(vif).sort_values(ascending=False)
ax.bar(s.index, s.values,
       color=["#d62828" if v > 10 else "#f77f00" if v > 5 else "#8b1a35" for v in s])
ax.axhline(10, ls="--", c="#d62828", label="VIF = 10 (severa)")
ax.axhline(5, ls=":", c="#f77f00", label="VIF = 5 (moderada)")
ax.set_ylabel("VIF"); ax.set_title("Caso 2 · Fase A — Factor de Inflacion de la Varianza")
plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8); ax.legend()
fig.tight_layout(); fig.savefig(f"{FIGDIR}/a_vif.png", dpi=115); plt.close(fig)


# =============================================================================
# FASE B
# =============================================================================
titulo("FASE B - PARTICION Y PREPROCESAMIENTO COMUN")
X, y = df[PREDICTORES], df[OBJETIVO]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
print(f"Entrenamiento: {len(X_tr)}   Prueba: {len(X_te)}   (80 / 20)")
print("\nAmbos modelos comparten el mismo preprocesamiento (StandardScaler);")
print("la UNICA diferencia es la expansion polinomica.")
res["n_train"], res["n_test"] = len(X_tr), len(X_te)


def evaluar(nombre, grado):
    pasos = [("sc", StandardScaler())]
    if grado > 1:
        pasos.insert(0, ("poly", PolynomialFeatures(degree=grado, include_bias=False)))
    pipe = Pipeline(pasos + [("modelo", LinearRegression())])
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
        "rmse_test": round(float(np.sqrt(mean_squared_error(y_te, p_te))), 4),
        "mae_test": round(float(mean_absolute_error(y_te, p_te)), 4),
    }
    m["brecha"] = round(m["r2_train"] - m["r2_cv"], 4)
    print(f"  {nombre:<32} p={m['n_terminos']:>3}  R2tr={m['r2_train']:.4f}  "
          f"R2te={m['r2_test']:.4f}  CV={m['r2_cv']:.4f}+-{m['r2_cv_std']:.3f}  "
          f"RMSE={m['rmse_test']:.4f}  MAE={m['mae_test']:.4f}")
    return m, pipe


titulo("FASE B.1 - REGRESION LINEAL MULTIPLE")
m_lin, pipe_lin = evaluar("Lineal multiple (OLS)", 1)

print("\nCoeficientes (variables estandarizadas, por magnitud):")
co = pd.Series(pipe_lin.named_steps["modelo"].coef_, index=PREDICTORES)
for k, v in co.reindex(co.abs().sort_values(ascending=False).index).items():
    print(f"  {k:<24}{v:+.4f}")
res["coeficientes_lineal"] = {k: round(float(v), 4) for k, v in co.items()}

titulo("FASE B.2 - REGRESION POLINOMICA GRADO 2")
m_poly, pipe_poly = evaluar("Polinomica grado 2 (OLS)", 2)

titulo("FASE B.3 - COMPARATIVA")
tabla = pd.DataFrame([m_lin, m_poly]).set_index("nombre")
print(tabla[["n_terminos", "r2_train", "r2_test", "r2_cv", "brecha",
             "rmse_test", "mae_test"]].to_string())

d_r2 = m_poly["r2_test"] - m_lin["r2_test"]
print(f"\n  La expansion polinomica multiplica los terminos por "
      f"{m_poly['n_terminos']/m_lin['n_terminos']:.1f} "
      f"({m_lin['n_terminos']} -> {m_poly['n_terminos']}).")
print(f"  R2 de entrenamiento: {m_lin['r2_train']:.4f} -> {m_poly['r2_train']:.4f} "
      f"({m_poly['r2_train']-m_lin['r2_train']:+.4f})  MEJORA aparente")
print(f"  R2 de prueba       : {m_lin['r2_test']:.4f} -> {m_poly['r2_test']:.4f} "
      f"({d_r2:+.4f})  EMPEORA")
print(f"  Brecha train-CV    : {m_lin['brecha']:.4f} -> {m_poly['brecha']:.4f}")
print("\n  Sobreajuste de manual: el modelo memoriza el entrenamiento y generaliza")
print("  peor. Se recomienda la REGRESION LINEAL MULTIPLE para este dominio.")
res["fase_b"] = {"lineal": m_lin, "polinomica": m_poly, "delta_r2_test": round(d_r2, 4)}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
et = ["Lineal múltiple\n(%d términos)" % m_lin["n_terminos"],
      "Polinómica grado 2\n(%d términos)" % m_poly["n_terminos"]]
x = np.arange(2); an = 0.35
axes[0].bar(x - an/2, [m_lin["r2_train"], m_poly["r2_train"]], an,
            label="Entrenamiento", color="#8b1a35")
axes[0].bar(x + an/2, [m_lin["r2_test"], m_poly["r2_test"]], an,
            label="Prueba", color="#f77f00")
axes[0].set_xticks(x); axes[0].set_xticklabels(et); axes[0].set_ylabel("R²")
axes[0].set_title("Caso 2 · Fase B — El sobreajuste, visible")
axes[0].legend()
for i, v in enumerate([m_lin["r2_train"], m_poly["r2_train"]]):
    axes[0].annotate(f"{v:.4f}", (i - an/2, v), ha="center", va="bottom", fontsize=9)
for i, v in enumerate([m_lin["r2_test"], m_poly["r2_test"]]):
    axes[0].annotate(f"{v:.4f}", (i + an/2, v), ha="center", va="bottom", fontsize=9)

co_ord = co.reindex(co.abs().sort_values().index)
axes[1].barh(co_ord.index, co_ord.values,
             color=["#d62828" if v < 0 else "#1d9e6f" for v in co_ord])
axes[1].set_xlabel("Coeficiente (variables estandarizadas)")
axes[1].set_title("Caso 2 · Fase B — Peso de cada propiedad (lineal múltiple)")
axes[1].tick_params(labelsize=8)
fig.tight_layout(); fig.savefig(f"{FIGDIR}/b_comparativa.png", dpi=115); plt.close(fig)

with open("resultados_fases_a_b.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)

titulo("ARCHIVOS GENERADOS")
print("  resultados_fases_a_b.json")
print(f"  {FIGDIR}/a_outliers.png · a_correlacion.png · a_vif.png · b_comparativa.png")
