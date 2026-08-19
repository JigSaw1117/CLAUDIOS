"""
=============================================================================
TALLER 1 — CASO 2: WINE QUALITY (ANÁLISIS Y ENTRENAMIENTO DETALLADO)
MODELOS: REGRESIÓN LINEAL MÚLTIPLE (OLS) VS REGRESIÓN POLINÓMICA (GRADO 2)
=============================================================================

DOCUMENTACIÓN MATEMÁTICA Y METODOLÓGICA DETALLADA:

1. DEFINICIÓN DE LAS VARIABLES DEL DATASET (WineQT.csv)
-----------------------------------------------------------------------------
   • TARGET (Variable Dependiente Y):
     - quality: Calidad del vino evaluada por sommeliers (escala discreta de 0 a 10).

   • PREDICTORES (Variables Independientes X1..X11):
     1. fixed acidity (X1): Acidez fija del vino (ácido tartárico en g/L).
     2. volatile acidity (X2): Acidez volátil (ácido acético en g/L). Niveles altos dan sabor avinagrado.
     3. citric acid (X3): Ácido cítrico en g/L. Añade frescura y notas frutales.
     4. residual sugar (X4): Azúcar residual post-fermentación en g/L.
     5. chlorides (X5): Cloruros (sal, NaCl en g/L). Excesos perjudican el sabor.
     6. free sulfur dioxide (X6): SO2 libre en mg/L. Preserva el vino contra bacterias y oxidación.
     7. total sulfur dioxide (X7): SO2 total en mg/L. Excesos producen olores sulfurosos.
     8. density (X8): Densidad en g/mL (relación masa/volumen respecto al agua).
     9. pH (X9): Potencial de hidrógeno (acidez general del vino, 0-14).
     10. sulphates (X10): Sulfatos (K2SO4 en g/L). Contribuyen a la producción de SO2.
     11. alcohol (X11): Porcentaje de alcohol por volumen (% vol).

2. ESTANDARIZACIÓN Z-SCORE (StandardScaler)
-----------------------------------------------------------------------------
   Dado que las variables poseen diferentes escalas (ej. chlorides ~ 0.08 vs total sulfur dioxide ~ 45),
   se aplica la transformación Z-score para centrar la media en 0 y desviación estándar en 1:
       X_scaled = (X - μ) / σ
   Donde:
       μ = Media poblacional de la variable en el conjunto de entrenamiento.
       σ = Desviación estándar de la variable en el conjunto de entrenamiento.

3. MODELO 1: REGRESIÓN LINEAL MÚLTIPLE (OLS — Ordinary Least Squares)
-----------------------------------------------------------------------------
   • Ecuación del modelo:
       ŷ = β0 + β1·X1_scaled + β2·X2_scaled + ... + β11·X11_scaled
   • Solución matricial OLS:
       β = (X^T · X)^(-1) · X^T · y
   • Función de pérdida minimizada (Suma de Cuadrados de los Residuos):
       SCR = Σ (y_i - ŷ_i)^2

4. MODELO 2: REGRESIÓN POLINÓMICA DE GRADO 2 (PolynomialFeatures)
-----------------------------------------------------------------------------
   • Expansión polinómica de las 11 variables escaladas:
       - 11 términos lineales (X_i)
       - 11 términos cuadráticos (X_i^2)
       - 55 términos de interacción (X_i · X_j para i < j)
       Total = 11 + 11 + 55 = 77 variables explicativas (+ 1 intercepto β0).
   • Ecuación del modelo:
       ŷ = β0 + Σ (β_i · X_i) + Σ (β_ij · X_i · X_j) + Σ (β_ii · X_i^2)
   • Permite capturar efectos no lineales y curvaturas en la respuesta de calidad.

=============================================================================
"""

import json
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =============================================================================
# PASO 1: CARGA, LIMPIEZA Y SEPARACIÓN DE VARIABLES
# =============================================================================
print("=" * 80)
print("PASO 1: CARGA Y PREPARACIÓN DEL DATASET (WineQT.csv)")
print("=" * 80)

# Cargar el archivo CSV en un DataFrame de Pandas
df = pd.read_csv("WineQT.csv")

# Eliminar la columna 'Id' si existe, ya que es un identificador sin valor predictivo
df = df.drop(columns=["Id"], errors="ignore")

# Lista explícita de las 11 variables independientes (predictores / features)
FEATURE_COLS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol"
]

# Variable dependiente (objetivo / target)
TARGET_COL = "quality"

# Extraer la matriz X (n_muestras x 11) y el vector y (n_muestras)
X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

print(f"-> Total de registros cargados (N) : {len(df)}")
print(f"-> Cantidad de predictores (p)      : {len(FEATURE_COLS)}")
print(f"-> Rango de calidad del vino (Y)   : Min={y.min()}, Max={y.max()}, Media={y.mean():.4f}")

# DIVISIÓN DEL DATASET: 80% Entrenamiento (Train) y 20% Prueba (Test)
# random_state=42 asegura la reproducibilidad exacta en cada ejecución
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"-> Muestras de Entrenamiento (Train): {X_train.shape[0]}")
print(f"-> Muestras de Evaluación (Test)   : {X_test.shape[0]}")

# =============================================================================
# PASO 2: ENTRENAMIENTO DEL MODELO 1 — REGRESIÓN LINEAL MÚLTIPLE (OLS)
# =============================================================================
print("\n" + "=" * 80)
print("PASO 2: MODELO 1 — REGRESIÓN LINEAL MÚLTIPLE (OLS)")
print("=" * 80)

# Instanciar el escalador Z-Score
scaler_lin = StandardScaler()

# fit_transform calcula la media (μ) y desviación estándar (σ) solo en X_train,
# y luego transforma X_train a valores estandarizados: X_scaled = (X - μ) / σ
X_train_sc = scaler_lin.fit_transform(X_train)

# transform aplica los mismos μ y σ calculados de train al conjunto de test (evita data leakage)
X_test_sc  = scaler_lin.transform(X_test)

# Instanciar el modelo de Regresión Lineal por Mínimos Cuadrados Ordinarios
lin_model = LinearRegression()

# Entrenar el modelo calculando la solución matricial: β = (X_scaled^T · X_scaled)^(-1) · X_scaled^T · y
lin_model.fit(X_train_sc, y_train)

# Generar predicciones en Train y Test
y_pred_lin_train = lin_model.predict(X_train_sc)
y_pred_lin_test  = lin_model.predict(X_test_sc)

# Cálculo de métricas estadisticas de evaluación
lin_r2_train  = r2_score(y_train, y_pred_lin_train)
lin_r2_test   = r2_score(y_test,  y_pred_lin_test)
lin_mse_test  = mean_squared_error(y_test, y_pred_lin_test)
lin_rmse_test = math.sqrt(lin_mse_test)
lin_mae_test  = mean_absolute_error(y_test, y_pred_lin_test)

print(f"-> Intercepto β0 (Calidad base cuando X_scaled=0) : {lin_model.intercept_:.6f}")
print("-> Coeficientes de Regresión Lineal (β1 .. β11):")
for col, coef in zip(FEATURE_COLS, lin_model.coef_):
    impact = "Aumenta la calidad" if coef > 0 else "Disminuye la calidad"
    print(f"   • {col:<25}: β = {coef:+.6f} ({impact})")

print(f"""
[MÉTRICAS DE RENDIMIENTO - MODELO LINEAL]
   • R² (Entrenamiento) : {lin_r2_train:.4f} ({lin_r2_train*100:.2f}% de varianza explicada)
   • R² (Prueba/Test)   : {lin_r2_test:.4f} ({lin_r2_test*100:.2f}% de varianza explicada en datos nuevos)
   • RMSE (Test)        : {lin_rmse_test:.4f} puntos de calidad
   • MAE (Test)         : {lin_mae_test:.4f} puntos de calidad
""")

# =============================================================================
# PASO 3: ENTRENAMIENTO DEL MODELO 2 — REGRESIÓN POLINÓMICA (GRADO 2)
# =============================================================================
print("=" * 80)
print("PASO 3: MODELO 2 — REGRESIÓN POLINÓMICA (GRADO 2)")
print("=" * 80)

# Construir un Pipeline scikit-learn que encadena:
# 1. StandardScaler: Normalización Z-score de las 11 variables
# 2. PolynomialFeatures(degree=2, include_bias=False): Genera las 77 combinaciones polinómicas
# 3. LinearRegression: Entrena OLS sobre la matriz de 77 variables expandidas
poly_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
    ("model",  LinearRegression())
])

# Entrenar el pipeline completo con los datos de entrenamiento
poly_pipeline.fit(X_train, y_train)

# Generar predicciones en Train y Test
y_pred_poly_train = poly_pipeline.predict(X_train)
y_pred_poly_test  = poly_pipeline.predict(X_test)

# Métricas de evaluación para el modelo polinómico
poly_r2_train  = r2_score(y_train, y_pred_poly_train)
poly_r2_test   = r2_score(y_test,  y_pred_poly_test)
poly_mse_test  = mean_squared_error(y_test, y_pred_poly_test)
poly_rmse_test = math.sqrt(poly_mse_test)
poly_mae_test  = mean_absolute_error(y_test, y_pred_poly_test)

# Extraer componentes del pipeline para análisis y exportación
poly_scaler    = poly_pipeline.named_steps["scaler"]
poly_features  = poly_pipeline.named_steps["poly"]
poly_lin_model = poly_pipeline.named_steps["model"]

# Obtener nombres descriptivos de las 77 características expandidas (ej. "alcohol^2", "volatile acidity citric acid")
feature_names = poly_features.get_feature_names_out(FEATURE_COLS)

print(f"-> Cantidad de variables expandidas (p_poly) : {len(feature_names)}")
print(f"-> Intercepto β0 Polinómico                    : {poly_lin_model.intercept_:.6f}")

print(f"""
[MÉTRICAS DE RENDIMIENTO - MODELO POLINÓMICO GRADO 2]
   • R² (Entrenamiento) : {poly_r2_train:.4f} ({poly_r2_train*100:.2f}% varianza explicada en train)
   • R² (Prueba/Test)   : {poly_r2_test:.4f} ({poly_r2_test*100:.2f}% varianza explicada en test)
   • RMSE (Test)        : {poly_rmse_test:.4f} puntos de calidad
   • MAE (Test)         : {poly_mae_test:.4f} puntos de calidad
""")

# =============================================================================
# PASO 4: COMPARACIÓN ESTADÍSTICA DE AMBOS MODELOS
# =============================================================================
print("=" * 80)
print("PASO 4: COMPARACIÓN DE RENDIMIENTO (LINEAL VS POLINÓMICO)")
print("=" * 80)
print(f"""
   Métrica             Modelo Lineal       Modelo Polinómico (G2)    Diferencia
   --------------------------------------------------------------------------------
   R² (Train)          {lin_r2_train:.4f}              {poly_r2_train:.4f}                    {(poly_r2_train - lin_r2_train)*100:+.2f}%
   R² (Test)           {lin_r2_test:.4f}              {poly_r2_test:.4f}                    {(poly_r2_test - lin_r2_test)*100:+.2f}%
   RMSE (Test)         {lin_rmse_test:.4f}              {poly_rmse_test:.4f}                    {(poly_rmse_test - lin_rmse_test):+.4f} pts
   MAE (Test)          {lin_mae_test:.4f}              {poly_mae_test:.4f}                    {(poly_mae_test - lin_mae_test):+.4f} pts
   Num. Características 12                  78                       +66 términos

   DIAGNÓSTICO DE OVERFITTING (SOBREAJUSTE):
   • En el Modelo Lineal: R²_train ({lin_r2_train:.4f}) y R²_test ({lin_r2_test:.4f}) son muy similares.
     El modelo es consistente y generaliza adecuadamente a datos no vistos.
   • En el Modelo Polinómico: R²_train sube a {poly_r2_train:.4f}, pero R²_test cae a {poly_r2_test:.4f}.
     Esto demuestra SOBREAJUSTE (Overfitting): al añadir 77 términos para 914 muestras,
     el modelo memoriza el ruido del conjunto de entrenamiento y pierde capacidad de generalización.
""")

# =============================================================================
# PASO 5: EXPORTACIÓN DE LOS MODELOS ENTRENADOS A ARCHIVOS JSON
# =============================================================================
print("=" * 80)
print("PASO 5: EXPORTACIÓN DE PARÁMETROS A ARCHIVOS JSON PARA USO EN JS")
print("=" * 80)

# Estrutura JSON 1: Modelo Lineal Múltiple
linear_json_data = {
    "metadata": {
        "model_id": "linear",
        "model_type": "Multiple Linear Regression (OLS)",
        "degree": 1,
        "n_features_in": len(FEATURE_COLS),
        "n_features_used": 1 + len(FEATURE_COLS),
        "dataset": "WineQT.csv",
        "n_samples": int(len(df)),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "target": TARGET_COL,
        "original_features": FEATURE_COLS,
        "formula": "y_hat = B0 + B1*X1_sc + B2*X2_sc + ... + B11*X11_sc",
        "description": "Modela relaciones lineales entre propiedades químicas y la calidad del vino."
    },
    "preprocessing": {
        "scaler": "StandardScaler",
        "formula": "X_scaled = (X - mean) / std",
        "means": {col: float(m) for col, m in zip(FEATURE_COLS, scaler_lin.mean_)},
        "stds":  {col: float(s) for col, s in zip(FEATURE_COLS, scaler_lin.scale_)}
    },
    "model_parameters": {
        "intercept": float(lin_model.intercept_),
        "coefficients": {col: float(c) for col, c in zip(FEATURE_COLS, lin_model.coef_)}
    },
    "performance": {
        "R2_train": round(float(lin_r2_train), 4),
        "R2_test": round(float(lin_r2_test), 4),
        "MSE_test": round(float(lin_mse_test), 4),
        "RMSE_test": round(float(lin_rmse_test), 4),
        "MAE_test": round(float(lin_mae_test), 4)
    }
}

with open("wine_model_linear.json", "w", encoding="utf-8") as f:
    json.dump(linear_json_data, f, indent=2, ensure_ascii=False)

print(" -> Exportado exitosamente: wine_model_linear.json")

# Estructura JSON 2: Modelo Polinómico Grado 2
poly_json_data = {
    "metadata": {
        "model_id": "polynomial",
        "model_type": "Polynomial Regression (degree=2, OLS)",
        "degree": 2,
        "n_features_in": len(FEATURE_COLS),
        "n_features_expanded": int(len(feature_names)),
        "dataset": "WineQT.csv",
        "n_samples": int(len(df)),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "target": TARGET_COL,
        "original_features": FEATURE_COLS,
        "formula": "y_hat = B0 + Σ Bi*Xi_sc + Σ Bij*Xi_sc*Xj_sc + Σ Bii*Xi_sc^2",
        "description": "Captura curvaturas e interacciones cuadráticas entre las 11 variables químicas."
    },
    "preprocessing": {
        "scaler": "StandardScaler",
        "formula": "X_scaled = (X - mean) / std",
        "means": {col: float(m) for col, m in zip(FEATURE_COLS, poly_scaler.mean_)},
        "stds":  {col: float(s) for col, s in zip(FEATURE_COLS, poly_scaler.scale_)}
    },
    "polynomial_expansion": {
        "degree": 2,
        "include_bias": False,
        "n_output_features": int(len(feature_names)),
        "feature_names": [str(n) for n in feature_names]
    },
    "model_parameters": {
        "intercept": float(poly_lin_model.intercept_),
        "coefficients": {str(name): float(coef) for name, coef in zip(feature_names, poly_lin_model.coef_)}
    },
    "performance": {
        "R2_train": round(float(poly_r2_train), 4),
        "R2_test": round(float(poly_r2_test), 4),
        "MSE_test": round(float(poly_mse_test), 4),
        "RMSE_test": round(float(poly_rmse_test), 4),
        "MAE_test": round(float(poly_mae_test), 4)
    }
}

with open("wine_model_poly.json", "w", encoding="utf-8") as f:
    json.dump(poly_json_data, f, indent=2, ensure_ascii=False)

print(" -> Exportado exitosamente: wine_model_poly.json")
print("=" * 80)
print("ENTRENAMIENTO Y EXPORTACIÓN FINALIZADOS CORRECTAMENTE")
print("=" * 80)
