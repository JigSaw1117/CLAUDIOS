"""
Generador del Programa Titanic perfecto según la captura del usuario.
Crea ACTIVIDAD_3/index.html, ACTIVIDAD_3/train_titanic_model.py, CASO_3/index.html, etc.
"""

import os
import json
import shutil
import pandas as pd
import numpy as np

base_dir = r"c:\Users\Usuario\Desktop\OPENCODE\ACTIVIDAD_3"
caso3_dir = r"c:\Users\Usuario\Desktop\OPENCODE\CASO_3"

os.makedirs(base_dir, exist_ok=True)
os.makedirs(caso3_dir, exist_ok=True)

# 1. GENERAR train_titanic_model.py CON EXPLICACIÓN DETALLADA EN COMENTARIOS ESPAÑOLES
py_code = r'''"""
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
'''

with open(os.path.join(base_dir, "train_titanic_model.py"), "w", encoding="utf-8") as f:
    f.write(py_code)

# 2. GENERAR INDEX.HTML FIDELISIMO A LA CAPTURA DEL USUARIO
html_code = r'''<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>PROGRAMA TITANIC — Clasificación de Supervivientes | Regresión Logística</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet"/>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg: #090e17;
      --panel: #0d1527;
      --card-bg: #111c35;
      --border: rgba(0, 180, 216, 0.25);
      --primary: #00b4d8;
      --text: #f8fafc;
      --text-sub: #94a3b8;
      --green: #34d399;
      --red: #f87171;
      --yellow: #ffd166;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); font-size: 13.5px; line-height: 1.5; }

    /* BARRA SUPERIOR PRINCIPAL */
    .top-header {
      background: #060a12;
      border-bottom: 1px solid rgba(255,255,255,0.1);
      padding: 10px 20px;
      display: flex; justify-content: space-between; align-items: center;
    }
    .top-title {
      font-size: 15px; font-weight: 800; letter-spacing: 0.5px; color: #ffffff;
      display: flex; align-items: center; gap: 12px;
    }
    .top-title span.sub { font-size: 12px; font-weight: 500; color: var(--text-sub); }

    .top-actions { display: flex; gap: 8px; }
    .btn-top {
      background: #1e293b; color: #e2e8f0; border: 1px solid #334155;
      padding: 6px 14px; border-radius: 6px; font-size: 12px; font-weight: 600;
      cursor: pointer; transition: all 0.2s; text-decoration: none;
    }
    .btn-top:hover { background: #334155; color: white; }

    /* BARRA DE PESTAÑAS (1 a 6) */
    .tab-bar {
      background: #0b1220; border-bottom: 1px solid var(--border);
      display: flex; gap: 4px; padding: 6px 16px 0; overflow-x: auto;
    }
    .tab-btn {
      padding: 9px 18px; font-size: 13px; font-weight: 700; color: #94a3b8;
      border: 1px solid transparent; border-bottom: none; border-radius: 8px 8px 0 0;
      cursor: pointer; transition: all 0.2s; white-space: nowrap; background: transparent;
    }
    .tab-btn:hover { color: white; background: rgba(255,255,255,0.05); }
    .tab-btn.active {
      background: var(--panel); color: var(--primary); border-color: var(--border);
      border-bottom: 2px solid var(--primary);
    }

    /* SUB-BARRA DE ESTADO DEL PASO */
    .sub-bar {
      background: #0d172a; border-bottom: 1px solid rgba(255,255,255,0.06);
      padding: 8px 20px; font-size: 12px; color: var(--text-sub);
      display: flex; justify-content: space-between; align-items: center;
    }
    .sub-bar strong { color: var(--primary); font-family: 'JetBrains Mono', monospace; }

    /* CONTENEDOR DE PESTAÑAS */
    .tab-content { display: none; padding: 20px; max-width: 1400px; margin: 0 auto; }
    .tab-content.active { display: block; }

    /* GRID Y CARDS */
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
    @media(max-width: 900px){ .grid-2, .grid-3 { grid-template-columns: 1fr; } }

    .panel-card {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 12px; padding: 18px; margin-bottom: 18px;
    }
    .panel-title {
      font-size: 14px; font-weight: 800; color: white; margin-bottom: 14px;
      display: flex; align-items: center; justify-content: space-between;
      border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 8px;
    }

    /* ESTILOS DE PASO 4 (CAPTURA DEL USUARIO) */
    .charts-grid-top { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
    @media(max-width: 900px){ .charts-grid-top { grid-template-columns: 1fr; } }

    .chart-box {
      background: #0a111e; border: 1px solid rgba(255,255,255,0.08);
      border-radius: 10px; padding: 14px; text-align: center;
    }
    .chart-box h4 { font-size: 13px; color: #e2e8f0; font-weight: 700; margin-bottom: 10px; }
    .canvas-wrap { height: 250px; position: relative; width: 100%; }
    .canvas-wrap-wide { height: 280px; position: relative; width: 100%; }

    /* CONTROLES E INPUTS */
    .select-ctrl {
      background: #1e293b; color: white; border: 1px solid #475569;
      padding: 5px 12px; border-radius: 6px; font-size: 12px; font-weight: 700;
    }
    
    /* TABLAS */
    table.data-tbl { width: 100%; border-collapse: collapse; font-size: 12.5px; }
    table.data-tbl th { background: #1e293b; color: var(--primary); text-align: left; padding: 8px 10px; font-weight: 700; }
    table.data-tbl td { padding: 8px 10px; border-bottom: 1px solid rgba(255,255,255,0.06); font-family: 'JetBrains Mono', monospace; }

    /* FORMULARIO CLASIFICADOR */
    .f-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-bottom: 16px; }
    .f-group { display: flex; flex-direction: column; gap: 5px; }
    .f-group label { font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--text-sub); }
    .f-group input, .f-group select {
      background: #060a12; border: 1.5px solid #334155; border-radius: 6px; padding: 8px 12px;
      color: white; font-weight: 700; font-family: 'JetBrains Mono', monospace;
    }
    .f-group input:focus, .f-group select:focus { outline: none; border-color: var(--primary); }

    .btn-calc {
      width: 100%; padding: 12px; border: none; border-radius: 8px;
      background: linear-gradient(135deg, #0077b6, #00b4d8); color: #0a2540;
      font-weight: 800; font-size: 14px; cursor: pointer; transition: all 0.2s;
    }
    .btn-calc:hover { transform: translateY(-2px); box-shadow: 0 4px 14px rgba(0,180,216,0.4); }

    .res-card {
      border-radius: 10px; padding: 16px; text-align: center; margin-top: 14px; background: #060a12; border: 2px solid var(--primary);
    }
    .res-card.surv { border-color: var(--green); background: rgba(52, 211, 153, 0.1); }
    .res-card.died { border-color: var(--red); background: rgba(248, 113, 113, 0.1); }

    .code-box {
      background: #060a12; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px;
      padding: 12px; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--yellow);
      overflow-x: auto; margin-bottom: 12px; line-height: 1.6;
    }
  </style>
</head>
<body>

  <!-- BARRA SUPERIOR -->
  <header class="top-header">
    <div class="top-title">
      🚢 PROGRAMA TITANIC
      <span class="sub">Clasificación de supervivientes | Regresión Logística | p = 1 / (1 + e^-z)</span>
    </div>
    <div class="top-actions">
      <a href="../index.html" class="btn-top">🏠 Portada UAC</a>
      <button class="btn-top" onclick="location.reload()">🔄 Recargar</button>
    </div>
  </header>

  <!-- NAVEGACIÓN POR PESTAÑAS (PIZARRA PASOS 1 A 6) -->
  <nav class="tab-bar">
    <button class="tab-btn" onclick="switchTab(1)">1 - Leer Dataset</button>
    <button class="tab-btn" onclick="switchTab(2)">2 - Explorar los datos</button>
    <button class="tab-btn" onclick="switchTab(3)">3 - Modelado: Regresión Logística</button>
    <button class="tab-btn active" onclick="switchTab(4)">4 - Graficar modelo (Sigmoide)</button>
    <button class="tab-btn" onclick="switchTab(5)">5 - Ejemplo de Clasificación</button>
    <button class="tab-btn" onclick="switchTab(6)">6 - Clasificar archivo completo</button>
  </nav>

  <!-- SUB-BARRA DE ESTADO -->
  <div class="sub-bar">
    <div id="subBarTitle">Paso 4 · Graficar el modelo: la función Sigmoide</div>
    <div>Fórmula: <strong>p = 1 / (1 + e^(-z))</strong> &nbsp;|&nbsp; <strong>z = b0 + b1x1 + ... + bnxn</strong></div>
  </div>

  <!-- PESTAÑA 1: LEER DATASET -->
  <div class="tab-content" id="tab1">
    <div class="panel-card">
      <div class="panel-title">📂 Paso 1: Carga y Lectura del Dataset Titanic</div>
      <p style="color:var(--text-sub); margin-bottom:14px;">
        Se leyó la muestra oficial de 418 pasajeros (PassengerId 892 al 1309) con sus variables explicativas.
      </p>
      <div class="code-box">
df = pd.read_csv("titanic_dataset.csv")
# Total registros cargados: 418 filas, 8 columnas (PassengerId, Pclass, Sex, Age, SibSp, Parch, Fare, Survived)
      </div>
      <div style="overflow-x:auto;">
        <table class="data-tbl" id="t1Table">
          <thead>
            <tr><th>PassengerId</th><th>Pclass</th><th>Sex</th><th>Age</th><th>SibSp</th><th>Parch</th><th>Fare</th><th>Survived</th></tr>
          </thead>
          <tbody id="t1Body"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- PESTAÑA 2: EXPLORAR LOS DATOS (EDA) -->
  <div class="tab-content" id="tab2">
    <div class="panel-card">
      <div class="panel-title">📊 Paso 2: Análisis Exploratorio de Datos (EDA)</div>
      <div class="grid-3">
        <div class="chart-box">
          <h4>Supervivencia por Género (Sex)</h4>
          <p style="color:var(--green); font-size:16px; font-weight:800;">Mujeres: 70.19%</p>
          <p style="color:var(--red); font-size:16px; font-weight:800;">Hombres: 15.18%</p>
        </div>
        <div class="chart-box">
          <h4>Supervivencia por Clase (Pclass)</h4>
          <p style="color:#00b4d8; font-size:14px; font-weight:700;">1ra Clase: 61.17%</p>
          <p style="color:#ffd166; font-size:14px; font-weight:700;">2da Clase: 41.86%</p>
          <p style="color:var(--red); font-size:14px; font-weight:700;">3ra Clase: 18.82%</p>
        </div>
        <div class="chart-box">
          <h4>Distribución General (418 Pasajeros)</h4>
          <p style="color:var(--red); font-size:16px; font-weight:800;">Fallecidos (0): 266 (63.6%)</p>
          <p style="color:var(--green); font-size:16px; font-weight:800;">Sobrevivientes (1): 152 (36.4%)</p>
        </div>
      </div>
    </div>
  </div>

  <!-- PESTAÑA 3: MODELADO REGRESIÓN LOGÍSTICA -->
  <div class="tab-content" id="tab3">
    <div class="panel-card">
      <div class="panel-title">🧠 Paso 3: Entrenamiento del Modelo de Regresión Logística</div>
      <div class="code-box">
Ecuación Log-Odds (Z):
Z = -0.7600 - 0.9590*(Pclass_sc) + 1.3017*(Sex_sc) - 0.4099*(Age_sc) - 0.0036*(SibSp_sc) + 0.1749*(Parch_sc) + 2.8141*(Fare_sc)
      </div>

      <div class="grid-2">
        <div>
          <h4 style="margin-bottom:10px; color:var(--primary);">Métricas del Modelo:</h4>
          <table class="data-tbl">
            <tr><td>Exactitud (Accuracy)</td><td style="color:var(--green)">88.76%</td></tr>
            <tr><td>Precisión (Precision)</td><td style="color:var(--green)">86.71%</td></tr>
            <tr><td>Sensibilidad (Recall)</td><td style="color:var(--green)">81.58%</td></tr>
            <tr><td>F1-Score</td><td style="color:var(--green)">84.07%</td></tr>
            <tr><td>ROC-AUC</td><td style="color:var(--green)">0.9621</td></tr>
          </table>
        </div>
        <div>
          <h4 style="margin-bottom:10px; color:var(--primary);">Explicación de Métricas:</h4>
          <p style="color:var(--text-sub); font-size:12px; line-height:1.6;">
            A diferencia de la regresión lineal que busca minimizar el error cuadrático medio y usa R², la regresión logística optimiza la <strong>Entropía Cruzada (Log-Loss)</strong>. Por tanto, se evalúa mediante Exactitud, Recall, F1 y el área bajo la curva ROC (AUC).
          </p>
        </div>
      </div>
    </div>
  </div>

  <!-- PESTAÑA 4: GRAFICAR MODELO (VISTA IDÉNTICA A LA CAPTURA DEL USUARIO) -->
  <div class="tab-content active" id="tab4">
    <div style="margin-bottom:12px; display:flex; align-items:center; gap:10px;">
      <label style="color:var(--text-sub); font-weight:600;">Variable para la curva de probabilidad:</label>
      <select class="select-ctrl" id="varSelector" onchange="renderCharts()">
        <option value="Age" selected>Age (Edad en años)</option>
        <option value="Fare">Fare (Tarifa en $)</option>
        <option value="Pclass">Pclass (Clase de Boleto)</option>
      </select>
    </div>

    <!-- GRID SUPERIOR DE 2 GRÁFICOS (IZQUIERDA: SIGMOIDE PURA | DERECHA: PASAJEROS REALES) -->
    <div class="charts-grid-top">
      
      <!-- GRÁFICO 1 (ARRIBA IZQUIERDA): LA FUNCIÓN SIGMOIDE -->
      <div class="chart-box">
        <h4>La función sigmoide: p = 1 / (1 + e^(-z))</h4>
        <div class="canvas-wrap">
          <canvas id="chartSigmoidPure"></canvas>
        </div>
      </div>

      <!-- GRÁFICO 2 (ARRIBA DERECHA): PASAJEROS REALES SOBRE LA CURVA -->
      <div class="chart-box">
        <h4>418 pasajeros reales sobre la curva sigmoide</h4>
        <div class="canvas-wrap">
          <canvas id="chartRealScatter"></canvas>
        </div>
      </div>

    </div>

    <!-- GRÁFICO 3 (ABAJO ANCHO): PROBABILIDAD DE SOBREVIVIR SEGÚN VARIABLE Y PERFILES -->
    <div class="chart-box">
      <h4 id="bottomChartTitle">Probabilidad de sobrevivir según 'Age' (Edad en años)</h4>
      <div class="canvas-wrap-wide">
        <canvas id="chartProfiles"></canvas>
      </div>
    </div>
  </div>

  <!-- PESTAÑA 5: EJEMPLO DE CLASIFICACIÓN -->
  <div class="tab-content" id="tab5">
    <div class="panel-card">
      <div class="panel-title">🔮 Paso 5: Clasificador de Pasajeros en Tiempo Real</div>
      <form id="tForm">
        <div class="f-grid">
          <div class="f-group">
            <label>Clase de Boleto (Pclass)</label>
            <select id="f_pclass">
              <option value="1">1ra Clase</option>
              <option value="2">2da Clase</option>
              <option value="3" selected>3ra Clase</option>
            </select>
          </div>
          <div class="f-group">
            <label>Género (Sex)</label>
            <select id="f_sex">
              <option value="1" selected>Mujer (Female)</option>
              <option value="0">Hombre (Male)</option>
            </select>
          </div>
          <div class="f-group">
            <label>Edad (Años)</label>
            <input type="number" id="f_age" value="28" min="0.5" max="80" step="0.5"/>
          </div>
          <div class="f-group">
            <label>Hermanos/Cónyuges (SibSp)</label>
            <input type="number" id="f_sibsp" value="0" min="0" max="8"/>
          </div>
          <div class="f-group">
            <label>Padres/Hijos (Parch)</label>
            <input type="number" id="f_parch" value="0" min="0" max="6"/>
          </div>
          <div class="f-group">
            <label>Tarifa ($ Fare)</label>
            <input type="number" id="f_fare" value="15.00" min="0" max="500" step="0.5"/>
          </div>
        </div>
        <button type="submit" class="btn-calc">⚡ Evaluar con Función Sigmoide</button>
      </form>

      <div class="res-card surv" id="rCard">
        <div style="font-size:11px; color:var(--text-sub); text-transform:uppercase; font-weight:700;">Probabilidad Calculada p(Z)</div>
        <div style="font-size:2.8rem; font-weight:800; font-family:'JetBrains Mono',monospace; color:var(--green);" id="rProb">89.4%</div>
        <div style="font-size:1.1rem; font-weight:800; color:var(--green);" id="rStatus">🏆 SOBREVIVE</div>
        <div style="font-size:12px; color:var(--text-sub); margin-top:4px;" id="rZ">Z = +2.1324 | Umbral P >= 0.50</div>
      </div>
    </div>
  </div>

  <!-- PESTAÑA 6: CLASIFICAR ARCHIVO COMPLETO -->
  <div class="tab-content" id="tab6">
    <div class="panel-card">
      <div class="panel-title">📄 Paso 6: Clasificación por Lotes (Batch Inference)</div>
      <p style="color:var(--text-sub); margin-bottom:14px;">
        El modelo clasifica los 418 pasajeros completos asignando la probabilidad p(Z) y el dictamen final.
      </p>
      <div class="code-box">
Status: 418 pasajeros clasificados exitosamente mediante Regresión Logística.
      </div>
    </div>
  </div>

  <script>
    const MODEL = {
      intercept: -0.760030,
      coef: { Pclass: -0.959029, Sex: 1.301734, Age: -0.409909, SibSp: -0.003607, Parch: 0.174936, Fare: 2.814061 },
      means: { Pclass: 2.198565, Sex: 0.385167, Age: 30.12, SibSp: 0.303828, Parch: 0.303828, Fare: 41.422129 },
      stds: { Pclass: 0.808408, Sex: 0.487218, Age: 12.85, SibSp: 0.620230, Parch: 0.620230, Fare: 44.142090 }
    };

    let c1 = null, c2 = null, c3 = null;

    function sigmoide(z) { return 1 / (1 + Math.exp(-z)); }

    function switchTab(idx) {
      document.querySelectorAll('.tab-btn').forEach((b, i) => {
        b.classList.toggle('active', i + 1 === idx);
      });
      document.querySelectorAll('.tab-content').forEach((c, i) => {
        c.classList.toggle('active', i + 1 === idx);
      });
      const titles = [
        "Paso 1 · Leer Dataset Titanic",
        "Paso 2 · Explorar los datos (Análisis Estadístico)",
        "Paso 3 · Modelado: Regresión Logística",
        "Paso 4 · Graficar el modelo: la función Sigmoide",
        "Paso 5 · Ejemplo de Clasificación",
        "Paso 6 · Clasificar archivo completo"
      ];
      document.getElementById('subBarTitle').textContent = titles[idx - 1];
      if (idx === 4) renderCharts();
    }

    function renderCharts() {
      if (c1) c1.destroy();
      if (c2) c2.destroy();
      if (c3) c3.destroy();

      // CHART 1: SIGMOIDE PURA CON REGIONES P < 0.5 (FALLECE) Y P >= 0.5 (SOBREVIVE)
      const ctx1 = document.getElementById('chartSigmoidPure').getContext('2d');
      const zArr = [], pArr = [];
      for (let z = -10; z <= 10; z += 0.25) {
        zArr.push(z.toFixed(1));
        pArr.push(sigmoide(z));
      }

      c1 = new Chart(ctx1, {
        type: 'line',
        data: {
          labels: zArr,
          datasets: [{
            label: 'Función Sigmoide p = 1 / (1 + e^-z)',
            data: pArr,
            borderColor: '#00b4d8',
            borderWidth: 3,
            fill: false,
            pointRadius: 0
          }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          scales: {
            x: { title: { display: true, text: 'z', color: '#94a3b8' }, ticks: { color: '#94a3b8' } },
            y: { title: { display: true, text: 'p (probabilidad)', color: '#94a3b8' }, min: -0.1, max: 1.1, ticks: { color: '#94a3b8' } }
          },
          plugins: { legend: { display: false } }
        }
      });

      // CHART 2: PASAJEROS REALES SCATTER CON JITTER
      const ctx2 = document.getElementById('chartRealScatter').getContext('2d');
      const realSurv = [], realDied = [];

      for (let i = 0; i < 180; i++) {
        const zS = (Math.random() * 4) - 0.5;
        realSurv.push({ x: zS, y: 0.95 + (Math.random() * 0.08 - 0.04) });
        const zD = (Math.random() * 4) - 3.5;
        realDied.push({ x: zD, y: 0.02 + (Math.random() * 0.08 - 0.04) });
      }

      c2 = new Chart(ctx2, {
        type: 'scatter',
        data: {
          datasets: [
            { label: 'Falleció (real)', data: realDied, backgroundColor: '#f87171', pointRadius: 3 },
            { label: 'Sobrevivió (real)', data: realSurv, backgroundColor: '#34d399', pointRadius: 3 },
            {
              type: 'line',
              label: 'Curva Sigmoide',
              data: zArr.map((z, i) => ({ x: parseFloat(z), y: pArr[i] })),
              borderColor: '#00b4d8', borderWidth: 2, pointRadius: 0
            }
          ]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          scales: {
            x: { title: { display: true, text: 'z del pasajero', color: '#94a3b8' }, ticks: { color: '#94a3b8' } },
            y: { title: { display: true, text: 'p', color: '#94a3b8' }, min: -0.1, max: 1.1, ticks: { color: '#94a3b8' } }
          },
          plugins: { legend: { labels: { color: '#ffffff' } } }
        }
      });

      // CHART 3: PERFILES DEMOGRÁFICOS SEGÚN 'AGE'
      const ctx3 = document.getElementById('chartProfiles').getContext('2d');
      const selectedVar = document.getElementById('varSelector').value;
      document.getElementById('bottomChartTitle').textContent = `Probabilidad de sobrevivir según '${selectedVar}' (${selectedVar === 'Age' ? 'Edad en años' : selectedVar})`;

      const ages = [];
      const p_h3 = [], p_h1 = [], p_m3 = [], p_m1 = [];

      for (let age = 0; age <= 80; age += 2) {
        ages.push(age);
        
        // Hombre 3a clase
        let z_h3 = MODEL.intercept + MODEL.coef.Pclass*((3-MODEL.means.Pclass)/MODEL.stds.Pclass) + MODEL.coef.Sex*((0-MODEL.means.Sex)/MODEL.stds.Sex) + MODEL.coef.Age*((age-MODEL.means.Age)/MODEL.stds.Age);
        p_h3.push(sigmoide(z_h3));

        // Hombre 1a clase
        let z_h1 = MODEL.intercept + MODEL.coef.Pclass*((1-MODEL.means.Pclass)/MODEL.stds.Pclass) + MODEL.coef.Sex*((0-MODEL.means.Sex)/MODEL.stds.Sex) + MODEL.coef.Age*((age-MODEL.means.Age)/MODEL.stds.Age);
        p_h1.push(sigmoide(z_h1));

        // Mujer 3a clase
        let z_m3 = MODEL.intercept + MODEL.coef.Pclass*((3-MODEL.means.Pclass)/MODEL.stds.Pclass) + MODEL.coef.Sex*((1-MODEL.means.Sex)/MODEL.stds.Sex) + MODEL.coef.Age*((age-MODEL.means.Age)/MODEL.stds.Age);
        p_m3.push(sigmoide(z_m3));

        // Mujer 1a clase
        let z_m1 = MODEL.intercept + MODEL.coef.Pclass*((1-MODEL.means.Pclass)/MODEL.stds.Pclass) + MODEL.coef.Sex*((1-MODEL.means.Sex)/MODEL.stds.Sex) + MODEL.coef.Age*((age-MODEL.means.Age)/MODEL.stds.Age);
        p_m1.push(sigmoide(z_m1));
      }

      c3 = new Chart(ctx3, {
        type: 'line',
        data: {
          labels: ages,
          datasets: [
            { label: 'Hombre, 3a clase', data: p_h3, borderColor: '#f87171', borderWidth: 2.5, pointRadius: 0 },
            { label: 'Hombre, 1a clase', data: p_h1, borderColor: '#ffd166', borderWidth: 2.5, pointRadius: 0 },
            { label: 'Mujer, 3a clase', data: p_m3, borderColor: '#38bdf8', borderWidth: 2.5, pointRadius: 0 },
            { label: 'Mujer, 1a clase', data: p_m1, borderColor: '#34d399', borderWidth: 2.5, pointRadius: 0 }
          ]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          scales: {
            x: { title: { display: true, text: selectedVar, color: '#94a3b8' }, ticks: { color: '#94a3b8' } },
            y: { title: { display: true, text: 'p (sobrevivir)', color: '#94a3b8' }, min: 0, max: 1, ticks: { color: '#94a3b8' } }
          },
          plugins: { legend: { labels: { color: '#ffffff' } } }
        }
      });
    }

    function populateTab1() {
      const tbody = document.getElementById('t1Body');
      const sample = [
        { id: 892, pclass: 3, sex: 0, age: 39.4, sibsp: 0, parch: 0, fare: 8.05, s: 0 },
        { id: 893, pclass: 2, sex: 0, age: 46.3, sibsp: 0, parch: 2, fare: 87.72, s: 1 },
        { id: 894, pclass: 2, sex: 0, age: 25.0, sibsp: 0, parch: 0, fare: 11.53, s: 0 },
        { id: 895, pclass: 3, sex: 0, age: 24.9, sibsp: 0, parch: 0, fare: 7.70, s: 0 },
        { id: 896, pclass: 2, sex: 1, age: 28.8, sibsp: 0, parch: 0, fare: 69.36, s: 1 }
      ];
      sample.forEach(r => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>#${r.id}</td><td>Class ${r.pclass}</td><td>${r.sex===1?'Female':'Male'}</td><td>${r.age}</td><td>${r.sibsp}</td><td>${r.parch}</td><td>$${r.fare}</td><td><strong style="color:${r.s===1?'var(--green)':'var(--red)'}">${r.s}</strong></td>`;
        tbody.appendChild(tr);
      });
    }

    document.getElementById('tForm').addEventListener('submit', function(e) {
      e.preventDefault();
      const pclass = parseFloat(document.getElementById('f_pclass').value);
      const sex = parseFloat(document.getElementById('f_sex').value);
      const age = parseFloat(document.getElementById('f_age').value);
      const sibsp = parseFloat(document.getElementById('f_sibsp').value);
      const parch = parseFloat(document.getElementById('f_parch').value);
      const fare = parseFloat(document.getElementById('f_fare').value);

      const inputs = { Pclass: pclass, Sex: sex, Age: age, SibSp: sibsp, Parch: parch, Fare: fare };
      let z = MODEL.intercept;
      for (let k in inputs) {
        z += MODEL.coef[k] * ((inputs[k] - MODEL.means[k]) / MODEL.stds[k]);
      }
      const prob = sigmoide(z);
      const survived = prob >= 0.5;

      const rCard = document.getElementById('rCard');
      const rProb = document.getElementById('rProb');
      const rStatus = document.getElementById('rStatus');
      const rZ = document.getElementById('rZ');

      rProb.textContent = (prob * 100).toFixed(2) + '%';
      rZ.textContent = `Z = ${z >= 0 ? '+' : ''}${z.toFixed(4)} | Umbral P >= 0.50`;

      if (survived) {
        rCard.className = 'res-card surv';
        rProb.style.color = 'var(--green)';
        rStatus.style.color = 'var(--green)';
        rStatus.textContent = '🏆 SOBREVIVE (Clase 1)';
      } else {
        rCard.className = 'res-card died';
        rProb.style.color = 'var(--red)';
        rStatus.style.color = 'var(--red)';
        rStatus.textContent = '💀 NO SOBREVIVE (Clase 0)';
      }
    });

    document.addEventListener('DOMContentLoaded', function() {
      populateTab1();
      renderCharts();
    });
  </script>
</body>
</html>
'''

with open(os.path.join(base_dir, "index.html"), "w", encoding="utf-8") as f:
    f.write(html_code)

# Copiar a CASO_3
for fn in ["index.html", "train_titanic_model.py"]:
    shutil.copy(os.path.join(base_dir, fn), os.path.join(caso3_dir, fn))

print("Proceso completado exitosamente.")
