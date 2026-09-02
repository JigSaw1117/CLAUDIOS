"""
================================================================================
                    PROGRAMA TITANIC — REGRESIÓN LOGÍSTICA
================================================================================
EXPLICACIÓN DETALLADA DEL CÓDIGO Y LAS FÓRMULAS MATEMÁTICAS:

1. ¿CÓMO SE ENTRENÓ EL MODELO Y BAJO QUÉ FÓRMULA?
   A diferencia de la Regresión Lineal que utiliza el Error Cuadrático Medio y R²,
   la Regresión Logística para clasificación binaria utiliza la FUNCIÓN DE PÉRDIDA
   DE ENTROPÍA CRUZADA O LOG-LOSS (Logarithmic Loss):
   
       L(beta) = - (1/N) * sum [ y_i * ln(p_i) + (1 - y_i) * ln(1 - p_i) ]

   Donde:
   - y_i es la clase real (1 = Sobrevivió, 0 = Falleció)
   - p_i = sigma(Z_i) es la probabilidad predicha por la función Sigmoide.
   - N es el total de muestras (418 pasajeros).

2. ¿CUÁLES SON LAS FÓRMULAS DEL MODELO?
   a) Estandarización de características (Z-Score):
      X_scaled = (X - mu) / sigma
   
   b) Ecuación del Log-Odds (Z):
      Z = beta_0 + beta_1 * X_1 + beta_2 * X_2 + ... + beta_n * X_n
   
   c) Función Sigmoide (Mapeo a Probabilidad [0, 1]):
      p(Z) = 1 / (1 + e^(-Z))

   d) Regla de Decisión (Umbral 0.50):
      Si p >= 0.50 ==> Clasificación: 1 (Sobrevive)
      Si p < 0.50  ==> Clasificación: 0 (Fallece)

3. ¿POR QUÉ NO SE USA R² EN REGRESIÓN LOGÍSTICA?
   El coeficiente de determinación R² mide la varianza explicada en variables continuas.
   En clasificación (0 o 1), las métricas correctas son:
   - Exactitud (Accuracy): Porcentaje total de aciertos.
   - Precisión (Precision): Proporción de verdaderos sobrevivientes entre los predichos positivos.
   - Sensibilidad (Recall): Capacidad del modelo para detectar a todos los sobrevivientes reales.
   - Puntuación F1: Media armónica entre Precisión y Recall.
   - ROC-AUC: Capacidad de discriminación global del modelo (0.5 a 1.0).

================================================================================
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

print("================================================================================")
print("     PROGRAMA TITANIC: CLASIFICACIÓN DE SUPERVIVIENTES (REGRESIÓN LOGÍSTICA)    ")
print("================================================================================")

# ==============================================================================
# PASO 1: LEER DATASET TITANIC
# ==============================================================================
print("\n[PASO 1] Leer DataSet Titanic...")
csv_path = "titanic_dataset.csv"
if not os.path.exists(csv_path):
    csv_path = os.path.join(os.path.dirname(__file__), "titanic_dataset.csv")

df = pd.read_csv(csv_path)
print(f"-> Dataset cargado correctamente: {len(df)} filas (pasajeros), {len(df.columns)} columnas.")

# ==============================================================================
# PASO 2: EXPLORAR LOS DATOS (EDA)
# ==============================================================================
print("\n[PASO 2] Explorar los datos...")
print("-> Primeras 5 filas del dataset:")
print(df.head())

print("\n-> Estadísticas descriptivas generales:")
print(df.describe())

print("\n-> Distribución de la variable objetivo 'Survived':")
print(df["Survived"].value_counts().rename({0: "Falleció (0)", 1: "Sobrevivió (1)"}))

print("\n-> Tasa de supervivencia según Género (Sex: 1=Mujer, 0=Hombre):")
print(df.groupby("Sex")["Survived"].mean())

print("\n-> Tasa de supervivencia según Clase de Boleto (Pclass: 1, 2, 3):")
print(df.groupby("Pclass")["Survived"].mean())

# ==============================================================================
# PASO 3: MODELADO : REGRESIÓN LOGÍSTICA
# ==============================================================================
print("\n[PASO 3] Modelado: Regresión Logística...")
feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
X = df[feature_cols].values
y = df["Survived"].values

# 3.1 Estandarización de las 6 variables explicativas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3.2 Ajuste del modelo optimizando la Log-Loss mediante el algoritmo L-BFGS
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_scaled, y)

# 3.3 Generación de probabilidades p(Z) y predicciones binarias y_pred
y_pred = model.predict(X_scaled)
y_prob = model.predict_proba(X_scaled)[:, 1]

# 3.4 Cálculo de métricas de clasificación
acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
auc = roc_auc_score(y, y_prob)
cm = confusion_matrix(y, y_pred).tolist()

print(f"-> Coeficiente Intercepto (Beta 0): {model.intercept_[0]:.6f}")
print("-> Coeficientes de Inclinación (Betas i):")
for col, coef in zip(feature_cols, model.coef_[0]):
    print(f"   - {col:10s}: {coef:+.6f}")

print("\n-> Evaluación del Rendimiento del Modelo:")
print(f"   * Exactitud (Accuracy) : {acc*100:.2f}% (Porcentaje total de decisiones correctas)")
print(f"   * Precisión (Precision): {prec*100:.2f}% (Acierto cuando el modelo predice Sobrevivió)")
print(f"   * Sensibilidad (Recall): {rec*100:.2f}% (Capacidad de encontrar a los sobrevivientes reales)")
print(f"   * F1-Score             : {f1*100:.2f}% (Media armónica entre Precisión y Recall)")
print(f"   * Área bajo curva ROC  : {auc:.4f} (Excelente discriminación > 0.90)")

# ==============================================================================
# PASO 4: GRAFICAR MODELO (FUNCIÓN SIGMOIDE)
# ==============================================================================
print("\n[PASO 4] Graficar modelo (Función Sigmoide)...")

z_values = np.dot(X_scaled, model.coef_[0]) + model.intercept_[0]
z_range = np.linspace(-6, 6, 300)
sigmoid_curve = 1 / (1 + np.exp(-z_range))

plt.figure(figsize=(11, 6), dpi=120)
plt.style.use('dark_background')

plt.plot(z_range, sigmoid_curve, color='#00b4d8', linewidth=3, label=r'Función Sigmoide $p(z) = \frac{1}{1 + e^{-z}}$')
plt.axhline(0.5, color='#ffd166', linestyle='--', linewidth=1.5, label='Umbral de Decisión (P = 0.50)')
plt.axvline(0.0, color='#ffffff', linestyle=':', linewidth=1.2, label='Límite Log-Odds (Z = 0)')

plt.scatter(z_values[y == 1], y_prob[y == 1], color='#34d399', alpha=0.7, edgecolors='none', s=40, label='Sobrevivió (Y=1 real)')
plt.scatter(z_values[y == 0], y_prob[y == 0], color='#f87171', alpha=0.6, edgecolors='none', s=40, label='Falleció (Y=0 real)')

plt.title('Regresión Logística — Curva Sigmoide del Titanic (Caso 3)', fontsize=14, fontweight='bold', pad=15, color='#ffffff')
plt.xlabel(r'Log-Odds $Z = \beta_0 + \sum \beta_i X_i$', fontsize=12, color='#cbd5e1')
plt.ylabel('Probabilidad p(Sobrevivir)', fontsize=12, color='#cbd5e1')
plt.grid(True, linestyle=':', alpha=0.3)
plt.legend(loc='upper left', frameon=True, facecolor='#0a2540', edgecolor='#00b4d8')
plt.tight_layout()

graph_path = "grafico_sigmoide.png"
plt.savefig(graph_path, dpi=200)
plt.close()
print(f"-> Gráfica guardada exitosamente como: {graph_path}")

# ==============================================================================
# PASO 5: EJEMPLO DE CLASIFICACIÓN
# ==============================================================================
print("\n[PASO 5] Ejemplo de Clasificación de Pasajeros Específicos...")

ejemplos = [
    {"nombre": "Pasajero A (Mujer 1ra Clase, 22 años, $80)", "input": [1, 1, 22.0, 1, 0, 80.0]},
    {"nombre": "Pasajero B (Hombre 3ra Clase, 30 años, $8.05)", "input": [3, 0, 30.0, 0, 0, 8.05]}
]

for ej in ejemplos:
    vals = np.array(ej["input"]).reshape(1, -1)
    vals_scaled = scaler.transform(vals)
    z_val = float(np.dot(vals_scaled, model.coef_[0])[0] + model.intercept_[0])
    prob = float(1 / (1 + np.exp(-z_val)))
    pred_cls = 1 if prob >= 0.5 else 0
    res_str = "SOBREVIVE (Clase 1)" if pred_cls == 1 else "NO SOBREVIVE (Clase 0)"
    
    print(f"\n* {ej['nombre']}:")
    print(f"  - Ecuación Z (Log-Odds) = {z_val:+.4f}")
    print(f"  - Sigmoide p(Z)         = {prob:.4f} ({prob*100:.2f}%)")
    print(f"  - Dictamen Final       = {res_str}")

# Exportar modelo JSON
export_data = {
    "model_name": "Logistic Regression Titanic Model",
    "intercept": float(model.intercept_[0]),
    "features": feature_cols,
    "coef": {col: float(c) for col, c in zip(feature_cols, model.coef_[0])},
    "means": {col: float(m) for col, m in zip(feature_cols, scaler.mean_)},
    "stds": {col: float(s) for col, s in zip(feature_cols, scaler.scale_)},
    "metrics": {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": cm
    }
}

json_path = "modelo_titanic.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(export_data, f, indent=2)

print(f"\n-> Modelo exportado en formato JSON a: {json_path}")
print("\n================================================================================")
print("                                FIN DEL PROGRAMA                                ")
print("================================================================================")
