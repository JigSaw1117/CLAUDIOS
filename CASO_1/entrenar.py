import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "housing.csv")
MODELO_PATH = os.path.join(BASE_DIR, "modelo_casas.pkl")
MODELO_JSON_PATH = os.path.join(BASE_DIR, "modelo.json")

COLUMNAS_NUMERICAS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
COLUMNA_CATEGORICA = "ocean_proximity"
COLUMNA_OBJETIVO = "median_house_value"


def main():
    df = pd.read_csv(CSV_PATH)

    modelo = LinearRegression()

    preprocesador = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ]
                ),
                COLUMNAS_NUMERICAS,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                [COLUMNA_CATEGORICA],
            ),
        ]
    )

    pipeline = Pipeline(steps=[("preprocesador", preprocesador), ("modelo", modelo)])

    X = df[COLUMNAS_NUMERICAS + [COLUMNA_CATEGORICA]]
    y = df[COLUMNA_OBJETIVO]

    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_entrenamiento, y_entrenamiento)

    predicciones = pipeline.predict(X_prueba)

    r2 = r2_score(y_prueba, predicciones)
    rmse = np.sqrt(mean_squared_error(y_prueba, predicciones))
    mae = mean_absolute_error(y_prueba, predicciones)

    print("=== REGRESION POLINOMICA (GRADO 2) - HOUSING ===")
    print(f"Filas de entrenamiento: {len(X_entrenamiento)}")
    print(f"Filas de prueba:        {len(X_prueba)}")
    print(f"R^2 (prueba):   {r2:.4f}")
    print(f"RMSE (prueba):  ${rmse:,.2f}")
    print(f"MAE (prueba):   ${mae:,.2f}")

    joblib.dump(
        {
            "pipeline": pipeline,
            "tipo_modelo": "Regresión Polinómica (Grado 2)",
            "grado": 2,
            "columnas_numericas": COLUMNAS_NUMERICAS,
            "columna_categorica": COLUMNA_CATEGORICA,
            "columna_objetivo": COLUMNA_OBJETIVO,
            "metricas": {"r2": r2, "rmse": rmse, "mae": mae},
        },
        MODELO_PATH,
    )
    print(f"Modelo guardado en: {MODELO_PATH}")

    _exportar_json(pipeline, r2, rmse, mae)


def _exportar_json(pipeline, r2, rmse, mae):
    lr = pipeline.named_steps["modelo"]
    ct = pipeline.named_steps["preprocesador"]
    num_step = ct.named_transformers_["num"]
    escalador = num_step.named_steps["scaler"]
    poly = num_step.named_steps["poly"]
    codificador = ct.named_transformers_["cat"]

    datos_modelo = {
        "tipo_modelo": "Regresión Polinómica (Grado 2)",
        "grado": 2,
        "columnas_numericas": COLUMNAS_NUMERICAS,
        "escalado": {
            "mean": escalador.mean_.round(6).tolist(),
            "scale": escalador.scale_.round(6).tolist(),
        },
        "powers": poly.powers_.tolist(),
        "categorias_oceano": codificador.categories_[0].tolist(),
        "coeficientes": [round(c, 8) for c in lr.coef_.tolist()],
        "intercepto": round(float(lr.intercept_), 8),
        "metricas": {
            "r2": round(float(r2), 4),
            "rmse": round(float(rmse), 2),
            "mae": round(float(mae), 2),
        },
    }

    with open(MODELO_JSON_PATH, "w", encoding="utf-8") as archivo:
        json.dump(datos_modelo, archivo, indent=2, ensure_ascii=False)

    print(f"Modelo web guardado en: {MODELO_JSON_PATH}")


if __name__ == "__main__":
    main()