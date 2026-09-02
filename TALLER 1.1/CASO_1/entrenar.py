"""
CASO 1 - CALIFORNIA HOUSING  |  REGRESION POLINOMICA DE GRADO 2
Universidad Andina del Cusco - Inteligencia Artificial (2026-II)

Entrena el modelo que consume el aplicativo web del Caso 1 y lo exporta en dos
formatos: modelo_casas.pkl (interfaz de escritorio) y modelo.json (pagina web,
que rehace la prediccion en JavaScript).

FUNCION OBJETIVO -- Minimos Cuadrados Ordinarios (OLS), sin regularizacion:

    J(b) = min  SUM (y_j - y_est_j)^2   =   || y - X*b ||^2
            b

    Solucion cerrada (ecuaciones normales):   b = (X^T X)^-1 X^T y

ECUACION DEL MODELO -- 49 coeficientes mas intercepto:

    y_est = b0 + SUM b_i*z_i + SUM b_ii*z_i^2 + SUM b_ij*z_i*z_j + SUM g_k*d_k
                 [8 lineales]  [8 cuadraticos]  [28 cruzados]      [5 dummies]

    donde   z_i = (x_i - mu_i) / sigma_i    variable estandarizada
            d_k = 1 si la fila pertenece a la categoria k, 0 en caso contrario

PARTICION: 80 % entrenamiento / 20 % prueba, con random_state=42.

Ejecutar:  python entrenar.py
"""

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

# Rutas absolutas respecto a este archivo: el script funciona sea cual sea
# el directorio de trabajo desde el que se lance.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "housing.csv")
MODELO_PATH = os.path.join(BASE_DIR, "modelo_casas.pkl")
MODELO_JSON_PATH = os.path.join(BASE_DIR, "modelo.json")

# Las 8 variables numericas (x_1 .. x_8) sobre las que se aplica la expansion
# polinomica. El ORDEN importa: es el que se guarda en modelo.json y el que
# espera el JavaScript de index.html para reconstruir la prediccion.
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
# Unica variable categorica: OneHotEncoder la convierte en 5 dummies.
COLUMNA_CATEGORICA = "ocean_proximity"
# Variable objetivo (y): valor mediano de vivienda del distrito, en US$.
COLUMNA_OBJETIVO = "median_house_value"


def main():
    # ------------------------------------------------------------- CARGA
    df = pd.read_csv(CSV_PATH)

    # ------------------------------------------------------ EL ESTIMADOR
    # LinearRegression resuelve b = (X^T X)^-1 X^T y por minimos cuadrados.
    # Es el MISMO estimador que usaria una regresion multiple: lo unico que
    # cambia es cuantas columnas tiene X tras el preprocesamiento.
    modelo = LinearRegression()

    # ---------------------------------------- FASE A: PREPROCESAMIENTO
    # Cada paso equivale a una parte de la formula. Va DENTRO del pipeline
    # a proposito: asi la mediana, la media y la desviacion se calculan solo
    # con los datos de entrenamiento y no se filtra informacion del conjunto
    # de prueba (data leakage).
    preprocesador = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        # Imputa los 207 nulos de total_bedrooms con la mediana.
                        ("imputer", SimpleImputer(strategy="median")),
                        # Estandariza:   z_i = (x_i - mu_i) / sigma_i
                        ("scaler", StandardScaler()),
                        # Expande a grado 2. OJO al orden: se aplica DESPUES
                        # de escalar, asi que los terminos son productos de
                        # variables ya estandarizadas (z_i*z_j), no productos
                        # estandarizados despues. 8 vars -> C(10,2)-1 = 44.
                        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ]
                ),
                COLUMNAS_NUMERICAS,
            ),
            (
                "cat",
                # Convierte ocean_proximity en 5 dummies d_k (0/1).
                # No se estandarizan: entran directas al modelo.
                OneHotEncoder(handle_unknown="ignore"),
                [COLUMNA_CATEGORICA],
            ),
        ]
    )

    # Pipeline completo: preprocesamiento -> modelo. Al llamar a .fit() se
    # ejecuta toda la cadena; al llamar a .predict(), tambien.
    pipeline = Pipeline(steps=[("preprocesador", preprocesador), ("modelo", modelo)])

    # X = matriz de predictores (8 numericas + 1 categorica)
    # y = vector objetivo
    X = df[COLUMNAS_NUMERICAS + [COLUMNA_CATEGORICA]]
    y = df[COLUMNA_OBJETIVO]

    # =================================================================
    # PARTICION DE LOS DATOS:  80 % ENTRENAMIENTO  /  20 % PRUEBA
    # =================================================================
    #   test_size=0.2    -> 20 % apartado para prueba      ->  4 128 filas
    #   el 80 % restante -> entrenamiento                  -> 16 512 filas
    #                       NO se declara: train_test_split lo calcula como
    #                       complemento (equivale a train_size=0.8).
    #   random_state=42  -> fija el barajado. Sin esto, cada ejecucion daria
    #                       una particion distinta y las metricas no serian
    #                       reproducibles.
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =================================================================
    # ENTRENAMIENTO  --  solo con el 80 %
    # =================================================================
    # Aqui se resuelve la funcion objetivo: .fit() halla los 49 coeficientes
    # y el intercepto minimizando SUM (y - y_est)^2. El conjunto de prueba
    # no interviene: el modelo no lo ve.
    pipeline.fit(X_entrenamiento, y_entrenamiento)

    # =================================================================
    # EVALUACION  --  solo con el 20 % que el modelo nunca vio
    # =================================================================
    # Se aplica el modelo ya entrenado a datos nuevos. Comparar estas
    # predicciones con y_prueba es lo que mide si generaliza o si memorizo.
    predicciones = pipeline.predict(X_prueba)

    # Metricas sobre el conjunto de PRUEBA:
    #   R2   = 1 - SUM(y-y_est)^2 / SUM(y-media)^2   varianza explicada
    #   RMSE = raiz(promedio de (y-y_est)^2)         error tipico, en US$
    #   MAE  = promedio de |y - y_est|               error medio, en US$
    r2 = r2_score(y_prueba, predicciones)
    rmse = np.sqrt(mean_squared_error(y_prueba, predicciones))
    mae = mean_absolute_error(y_prueba, predicciones)

    print("=== REGRESION POLINOMICA (GRADO 2) - HOUSING ===")
    print(f"Filas de entrenamiento: {len(X_entrenamiento)}")
    print(f"Filas de prueba:        {len(X_prueba)}")
    print(f"R^2 (prueba):   {r2:.4f}")
    print(f"RMSE (prueba):  ${rmse:,.2f}")
    print(f"MAE (prueba):   ${mae:,.2f}")

    # ------------------------------------------ EXPORTACION (binaria)
    # Pipeline completo serializado, para la interfaz de escritorio.
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
    """Exporta el modelo a JSON para que el navegador prediga sin Python.

    Se guardan los cuatro ingredientes que necesita la formula:
        escalado.mean / scale  ->  mu_i y sigma_i   (para estandarizar)
        powers                 ->  como construir cada termino polinomico
        coeficientes           ->  los b_i
        intercepto             ->  b0
    Con eso index.html rehace  y_est = b0 + SUM b_i * t_i  en JavaScript.
    """
    # Se extraen los objetos YA AJUSTADOS de dentro del pipeline.
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
        # Matriz de exponentes 44 x 8: cada fila indica a que potencia va
        # cada variable en ese termino. Ej.: [1,1,0,0,0,0,0,0] -> z_1 * z_2
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