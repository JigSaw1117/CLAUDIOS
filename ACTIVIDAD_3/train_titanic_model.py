"""
Programa Titanic — Regresión Logística
Pasos Pizarra:
1. Leer DataSet Titanic
2. Explorar los datos
3. Modelado: Regresión Logística
4. Graficar modelo (Función Sigmoide)
5. Ejemplo de Clasificación
Fin
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

print("==================================================")
print("             PROGRAMA TITANIC -- IA                ")
print("==================================================")

# 1. Leer DataSet Titanic
print("\n[PASO 1] Leer DataSet Titanic...")
csv_path = "titanic_dataset.csv"
if not os.path.exists(csv_path):
    csv_path = os.path.join(os.path.dirname(__file__), "titanic_dataset.csv")

df = pd.read_csv(csv_path)
print(f"-> Dataset cargado con exito. Total registros: {len(df)} filas, {len(df.columns)} columnas.")

# 2. Explorar los datos
print("\n[PASO 2] Explorar los datos...")
print("-> Primeros 5 registros:")
print(df.head())

print("\n-> Estadisticas descriptivas:")
print(df.describe())

print("\n-> Conteo de Supervivencia (0 = Fallecido, 1 = Sobrevivio):")
print(df["Survived"].value_counts())

print("\n-> Tasa de Supervivencia por Genero (Sex: 1=Mujer, 0=Hombre):")
print(df.groupby("Sex")["Survived"].mean())

print("\n-> Tasa de Supervivencia por Clase (Pclass):")
print(df.groupby("Pclass")["Survived"].mean())

# 3. Modelado: Regresion Logistica
print("\n[PASO 3] Modelado: Regresion Logistica...")
feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
X = df[feature_cols].values
y = df["Survived"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_scaled, y)

y_pred = model.predict(X_scaled)
y_prob = model.predict_proba(X_scaled)[:, 1]

acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
auc = roc_auc_score(y, y_prob)
cm = confusion_matrix(y, y_pred).tolist()

print(f"-> Coeficiente Intercepto (B0): {model.intercept_[0]:.6f}")
print("-> Coeficientes Beta (Bi):")
for col, coef in zip(feature_cols, model.coef_[0]):
    print(f"   - {col:10s}: {coef:.6f}")

print("\n-> Metricas de Desempeno:")
print(f"   - Exactitud (Accuracy) : {acc*100:.2f}%")
print(f"   - Precision (Precision): {prec*100:.2f}%")
print(f"   - Sensibilidad (Recall): {rec*100:.2f}%")
print(f"   - Puntuacion F1        : {f1*100:.2f}%")
print(f"   - Area ROC (AUC)       : {auc:.4f}")

# 4. Graficar modelo (Funcion Sigmoide)
print("\n[PASO 4] Graficar modelo (Funcion Sigmoide)...")
z_values = np.dot(X_scaled, model.coef_[0]) + model.intercept_[0]
z_range = np.linspace(-6, 6, 300)
sigmoid_curve = 1 / (1 + np.exp(-z_range))

plt.figure(figsize=(10, 6), dpi=120)
plt.style.use('dark_background')

plt.plot(z_range, sigmoid_curve, color='#00b4d8', linewidth=3, label=r'Funcion Sigmoide $\sigma(z) = \frac{1}{1 + e^{-z}}$')
plt.axhline(0.5, color='#ffd166', linestyle='--', linewidth=1.5, label='Umbral de Decision (P = 0.5)')
plt.axvline(0.0, color='#ffd166', linestyle=':', linewidth=1.2, label='Limite de Decision (Z = 0)')

plt.scatter(z_values[y == 1], y_prob[y == 1], color='#34d399', alpha=0.7, edgecolors='none', s=45, label='Sobrevivio (Y=1)')
plt.scatter(z_values[y == 0], y_prob[y == 0], color='#f87171', alpha=0.6, edgecolors='none', s=45, label='No Sobrevivio (Y=0)')

plt.title('Regresion Logistica -- Curva Sigmoide del Titanic', fontsize=14, fontweight='bold', pad=15, color='#ffffff')
plt.xlabel(r'Log-Odds $Z = \beta_0 + \sum \beta_i X_i$', fontsize=12, color='#cbd5e1')
plt.ylabel('Probabilidad Estimada de Supervivencia P(Y=1)', fontsize=12, color='#cbd5e1')
plt.grid(True, linestyle=':', alpha=0.3)
plt.legend(loc='upper left', frameon=True, facecolor='#0a2540', edgecolor='#00b4d8')
plt.tight_layout()

graph_path = "grafico_sigmoide.png"
plt.savefig(graph_path, dpi=200)
plt.close()
print(f"-> Grafica guardada como: {graph_path}")

# 5. Ejemplo de Clasificacion
print("\n[PASO 5] Ejemplo de Clasificacion...")
ejemplos = [
    {"nombre": "Pasajero A (Mujer 1ra Clase)", "input": [1, 1, 22.0, 1, 0, 80.0]},
    {"nombre": "Pasajero B (Hombre 3ra Clase)", "input": [3, 0, 30.0, 0, 0, 8.05]}
]

for ej in ejemplos:
    vals = np.array(ej["input"]).reshape(1, -1)
    vals_scaled = scaler.transform(vals)
    z_val = float(np.dot(vals_scaled, model.coef_[0])[0] + model.intercept_[0])
    prob = float(1 / (1 + np.exp(-z_val)))
    pred_cls = 1 if prob >= 0.5 else 0
    res_str = "SOBREVIVE" if pred_cls == 1 else "NO SOBREVIVE"
    
    print(f"\n--- {ej['nombre']} ---")
    print(f"    Valores : Pclass={ej['input'][0]}, Sex={'Mujer' if ej['input'][1]==1 else 'Hombre'}, Age={ej['input'][2]}, Fare=${ej['input'][5]}")
    print(f"    Log-Odds (Z): {z_val:.4f}")
    print(f"    Sigmoide s(Z): {prob:.4f} ({prob*100:.2f}%)")
    print(f"    Resultado    : {res_str}")

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

print(f"\n-> Modelo JSON exportado exitosamente a: {json_path}")
print("\n==================================================")
print("                   FIN                            ")
print("==================================================")
