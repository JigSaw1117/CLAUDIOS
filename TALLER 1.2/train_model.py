"""
Taller 1.2 - Clasificacion Binaria con Regresion Logistica
Potabilidad del Agua

Fase II: EDA y preparacion de datos.
Fase III: Modelado con Regresion Logistica.
Fase IV: Evaluacion de metricas.

Ejecutar:  python train_model.py
Genera:    figuras/*.png, resultados_metricas.json, modelo_potabilidad.pkl
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
FIGDIR = "figuras"
os.makedirs(FIGDIR, exist_ok=True)
sns.set_theme(style="whitegrid")

resultados = {}


def titulo(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# =============================================================================
# FASE II - CARGA Y EXPLORACION
# =============================================================================
titulo("FASE II.1 - CARGA DEL DATASET")

df = pd.read_csv("water_potability.csv")
FEATURES = [c for c in df.columns if c != "Potability"]
TARGET = "Potability"

DESCRIPCION = {
    "ph": "pH del agua (0-14, ideal 6.5-8.5)",
    "Hardness": "Dureza - capacidad de precipitar jabon (mg/L)",
    "Solids": "Solidos disueltos totales (ppm)",
    "Chloramines": "Cloraminas - desinfectante residual (ppm)",
    "Sulfate": "Sulfatos disueltos (mg/L)",
    "Conductivity": "Conductividad electrica (uS/cm)",
    "Organic_carbon": "Carbono organico total (ppm)",
    "Trihalomethanes": "Trihalometanos - subproducto de cloracion (ug/L)",
    "Turbidity": "Turbidez - material en suspension (NTU)",
}

n_muestras, n_variables = df.shape
n_duplicados = int(df.duplicated().sum())
nulos_por_col = df[FEATURES].isna().sum().to_dict()
balance_clases = df[TARGET].value_counts().sort_index().to_dict()

print(f"Muestras: {n_muestras}  |  Variables: {n_variables - 1}  |  Duplicados: {n_duplicados}")
print("Nulos por columna:", nulos_por_col)
print("Balance de clases (Potability):", balance_clases)

resultados["fase_ii"] = {
    "n_muestras": n_muestras,
    "n_variables": len(FEATURES),
    "duplicados": n_duplicados,
    "nulos_por_columna": {k: int(v) for k, v in nulos_por_col.items()},
    "balance_clases": {str(k): int(v) for k, v in balance_clases.items()},
    "balance_clases_pct": {
        str(k): round(100 * v / n_muestras, 2) for k, v in balance_clases.items()
    },
    "estadisticos": df[FEATURES].describe().to_dict(),
}

# --- Grafica: distribucion de la clase objetivo -----------------------------
plt.figure(figsize=(5, 4))
ax = sns.countplot(x=TARGET, data=df, hue=TARGET, palette=["#e63946", "#2a9d8f"], legend=False)
ax.set_xticks([0, 1])
ax.set_xticklabels(["No potable (0)", "Potable (1)"])
ax.set_title("Distribucion de la clase objetivo")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/a_balance_clases.png", dpi=130)
plt.close()

# --- Grafica: nulos por columna ----------------------------------------------
plt.figure(figsize=(7, 4))
nulos_series = pd.Series(nulos_por_col).sort_values(ascending=False)
sns.barplot(x=nulos_series.values, y=nulos_series.index, hue=nulos_series.index,
            palette="rocket", legend=False)
plt.title("Valores nulos por variable")
plt.xlabel("Cantidad de nulos")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/a_nulos.png", dpi=130)
plt.close()

# --- Grafica: boxplots por variable (outliers) -------------------------------
fig, axes = plt.subplots(3, 3, figsize=(13, 10))
for ax, col in zip(axes.flat, FEATURES):
    sns.boxplot(y=df[col], ax=ax, color="#00b4d8")
    ax.set_title(col, fontsize=10)
    ax.set_ylabel("")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/a_boxplots.png", dpi=130)
plt.close()

# --- Grafica: histogramas por clase ------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(13, 10))
for ax, col in zip(axes.flat, FEATURES):
    sns.histplot(data=df, x=col, hue=TARGET, kde=True, ax=ax, palette=["#e63946", "#2a9d8f"],
                 legend=False, element="step", stat="density", common_norm=False)
    ax.set_title(col, fontsize=10)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/a_histogramas_por_clase.png", dpi=130)
plt.close()

# --- Grafica: correlacion -----------------------------------------------------
plt.figure(figsize=(8, 6.5))
corr = df[FEATURES + [TARGET]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
            cbar_kws={"shrink": 0.8})
plt.title("Matriz de correlacion")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/a_correlacion.png", dpi=130)
plt.close()

correlaciones_con_target = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
resultados["fase_ii"]["correlacion_con_target"] = correlaciones_con_target.round(4).to_dict()

# =============================================================================
# FASE II.2 - PREPARACION: IMPUTACION, ESCALADO, DIVISION TRAIN/TEST
# =============================================================================
titulo("FASE II.2 - PREPARACION DE DATOS")

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape[0]} muestras  |  Test: {X_test.shape[0]} muestras")
print("Balance train:", y_train.value_counts().to_dict())
print("Balance test:", y_test.value_counts().to_dict())

# El imputer y el scaler se ajustan solo con train (dentro del Pipeline, ver mas
# abajo) para no filtrar informacion del set de prueba. SimpleImputer(median)
# porque ph, Sulfate y Trihalomethanes tienen nulos y la mediana es robusta a
# los outliers que se ven en los boxplots.
preprocesador = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), FEATURES),
    ]
)

resultados["fase_ii"]["split"] = {
    "train_size": int(X_train.shape[0]),
    "test_size": int(X_test.shape[0]),
    "train_balance": {str(k): int(v) for k, v in y_train.value_counts().to_dict().items()},
    "test_balance": {str(k): int(v) for k, v in y_test.value_counts().to_dict().items()},
}

# =============================================================================
# FASE III - MODELADO: REGRESION LOGISTICA
# =============================================================================
titulo("FASE III - ENTRENAMIENTO DEL MODELO")

# class_weight="balanced" porque el desbalance 61%/39% (ver Fase II) hace que
# una regresion logistica sin ajustar prediga siempre la clase mayoritaria
# (Precision/Recall/F1 = 0 para la clase minoritaria, un clasificador inutil
# aunque su Accuracy parezca decente). Balanceando los pesos, el umbral 0.5
# efectivo se corrige y el modelo si logra detectar agua potable.
pipeline = Pipeline([
    ("preprocesador", preprocesador),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
])
pipeline.fit(X_train, y_train)

coefs = pipeline.named_steps["clf"].coef_[0]
intercept = float(pipeline.named_steps["clf"].intercept_[0])

tabla_coefs = (
    pd.Series(coefs, index=FEATURES)
    .sort_values(key=abs, ascending=False)
)
print("Intercepto:", round(intercept, 4))
print("Coeficientes (escala estandarizada, ordenados por influencia):")
print(tabla_coefs.round(4))

resultados["fase_iii"] = {
    "intercepto": intercept,
    "coeficientes": tabla_coefs.round(6).to_dict(),
    "hiperparametros": {"max_iter": 1000, "solver": "lbfgs", "penalty": "l2",
                         "class_weight": "balanced", "random_state": RANDOM_STATE},
}

plt.figure(figsize=(7, 5))
colores = ["#2a9d8f" if v > 0 else "#e63946" for v in tabla_coefs.values]
sns.barplot(x=tabla_coefs.values, y=tabla_coefs.index, hue=tabla_coefs.index,
            palette=colores, legend=False)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Coeficientes de la Regresion Logistica (escala estandarizada)")
plt.xlabel("Coeficiente")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/b_coeficientes.png", dpi=130)
plt.close()

# =============================================================================
# FASE IV - EVALUACION DE METRICAS (sobre el set de prueba)
# =============================================================================
titulo("FASE IV - EVALUACION DE METRICAS")

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

print(f"Matriz de confusion -> TN={tn}  FP={fp}  FN={fn}  TP={tp}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")

resultados["fase_iv"] = {
    "matriz_confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "auc_roc": auc,
}

# --- Grafica: matriz de confusion ---------------------------------------------
plt.figure(figsize=(5, 4.3))
cm = np.array([[tn, fp], [fn, tp]])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["No potable", "Potable"], yticklabels=["No potable", "Potable"])
plt.xlabel("Prediccion")
plt.ylabel("Real")
plt.title("Matriz de Confusion")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/c_matriz_confusion.png", dpi=130)
plt.close()

# --- Grafica: curva ROC ---------------------------------------------------------
plt.figure(figsize=(5.5, 5))
plt.plot(fpr, tpr, color="#0077b6", linewidth=2, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Azar (AUC = 0.5)")
plt.xlabel("Tasa de Falsos Positivos (FPR)")
plt.ylabel("Tasa de Verdaderos Positivos (TPR / Recall)")
plt.title("Curva ROC")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/c_curva_roc.png", dpi=130)
plt.close()

# =============================================================================
# FASE V - EXPORTACION DEL MODELO
# =============================================================================
titulo("FASE V - EXPORTACION DEL MODELO Y RESULTADOS")

joblib.dump(pipeline, "modelo_potabilidad.pkl")
print("Modelo guardado en modelo_potabilidad.pkl")

# --- Exportacion del modelo entrenado para consumo en el navegador ----------
# El aplicativo web (app_web/) NO reimplementa la formula con los coeficientes
# escritos a mano: carga este artefacto (parametros realmente ajustados con
# el pipeline sobre el set de entrenamiento) y un motor de inferencia
# generico (app_web/predictor.js) hace imputacion -> estandarizacion ->
# combinacion lineal -> sigmoide a partir de estos datos, sea cual sea el
# dataset o los coeficientes con los que se reentrene.
imputer_ajustado = pipeline.named_steps["preprocesador"].named_transformers_["num"].named_steps["imputer"]
scaler_ajustado = pipeline.named_steps["preprocesador"].named_transformers_["num"].named_steps["scaler"]

rangos = {c: [float(df[c].min()), float(df[c].max())] for c in FEATURES}
medianas = {c: float(df[c].median()) for c in FEATURES}

modelo_exportado = {
    "caso": "Taller 1.2 - Potabilidad del Agua | Regresion Logistica",
    "features": FEATURES,
    "descripcion": DESCRIPCION,
    "rangos_observados": rangos,
    "imputacion_mediana_train": {c: float(v) for c, v in zip(FEATURES, imputer_ajustado.statistics_)},
    "escalado_train": {
        "media": {c: float(v) for c, v in zip(FEATURES, scaler_ajustado.mean_)},
        "desviacion": {c: float(v) for c, v in zip(FEATURES, scaler_ajustado.scale_)},
    },
    "regresion_logistica": {
        "intercepto": intercept,
        "coeficientes": {c: float(v) for c, v in zip(FEATURES, coefs)},
        "umbral_decision": 0.5,
        "clases": {"0": "No potable", "1": "Potable"},
    },
    "metricas_test": resultados["fase_iv"],
}

os.makedirs("app_web", exist_ok=True)
with open("app_web/modelo_potabilidad.json", "w", encoding="utf-8") as f:
    json.dump(modelo_exportado, f, indent=2, ensure_ascii=False)

# Tambien como .js (misma info, asignada a una variable global) para que la
# pagina funcione abriendo el archivo directamente (file://), sin chocar con
# las restricciones CORS que un fetch() de un .json local si tendria.
with open("app_web/modelo_potabilidad.js", "w", encoding="utf-8") as f:
    f.write("// Generado por train_model.py - NO editar a mano.\n")
    f.write("// Parametros reales del pipeline (imputer + scaler + regresion logistica)\n")
    f.write("// ajustado sobre el set de entrenamiento. Ver train_model.py, Fase V.\n")
    f.write("const MODELO_POTABILIDAD = ")
    json.dump(modelo_exportado, f, indent=2, ensure_ascii=False)
    f.write(";\n")

print("Modelo exportado para el navegador en app_web/modelo_potabilidad.{json,js}")

resultados["fase_v"] = {
    "rangos_observados": rangos,
    "medianas": medianas,
    "archivo_modelo": "modelo_potabilidad.pkl",
    "archivo_modelo_web": "app_web/modelo_potabilidad.json",
}

with open("resultados_metricas.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)
print("Resultados guardados en resultados_metricas.json")

titulo("LISTO")
