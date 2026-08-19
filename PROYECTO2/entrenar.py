# -*- coding: utf-8 -*-
"""Entrena un modelo de regresion lineal multiple (con terminos polinomiales)
con housing.csv y guarda el resultado entrenado en modelo.json y modelo.js.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

RUTA_CSV = "archive/housing.csv"
RUTA_JSON = "modelo.json"
RUTA_JS = "modelo.js"

df = pd.read_csv(RUTA_CSV)

df = df.dropna()
df = df[df["median_house_value"] < 500001]

df["cuartos_por_casa"] = df["total_rooms"] / df["households"]
df["banyos_por_casa"] = df["total_bedrooms"] / df["households"]
df["poblacion_por_casa"] = df["population"] / df["households"]
df["ingreso_medio"] = df["median_income"].clip(upper=12)
df["antiguedad"] = df["housing_median_age"]
df["latitud"] = df["latitude"]
df["longitud"] = df["longitude"]
df = df[df["poblacion_por_casa"] < 20]

oc = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
etiquetas_mar = {
    "<1H OCEAN": "A menos de 1 h del océano",
    "INLAND": "Tierra adentro",
    "ISLAND": "Isla",
    "NEAR BAY": "Cerca de la bahía",
    "NEAR OCEAN": "Cerca del océano",
}

base = ["cuartos_por_casa", "banyos_por_casa", "poblacion_por_casa",
        "ingreso_medio", "antiguedad", "latitud", "longitud"]

X = pd.DataFrame({c: df[c] for c in base})
for nivel in oc:
    X[f"mar_{nivel}"] = (df["ocean_proximity"] == nivel).astype(int)

X = X.copy()
X["poblacion_por_casa"] = np.log1p(X["poblacion_por_casa"])

y = np.log1p(df["median_house_value"].values)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

medias = X_train.mean()
desv = X_train.std().replace(0, 1.0)

Z_train = (X_train - medias) / desv
Z_test = (X_test - medias) / desv

polinomio = PolynomialFeatures(2, include_bias=False)
P_train = polinomio.fit_transform(Z_train)
P_test = polinomio.transform(Z_test)

modelo = LinearRegression()
modelo.fit(P_train, y_train)

def en_usd(valores_log):
    return np.expm1(valores_log)

pred_train = en_usd(modelo.predict(P_train))
pred_test = en_usd(modelo.predict(P_test))

r2_train = r2_score(en_usd(y_train), pred_train)
r2_test = r2_score(en_usd(y_test), pred_test)
rmse_usd = float(np.sqrt(mean_squared_error(en_usd(y_test), pred_test)))
rmse_log = float(np.sqrt(mean_squared_error(y_test, modelo.predict(P_test))))

num_meta = {
    "cuartos_por_casa": {"etiqueta": "Número de cuartos (por casa)", "min": 1, "max": 8, "paso": 0.5, "unidad": ""},
    "banyos_por_casa": {"etiqueta": "Número de baños (por casa)", "min": 1, "max": 4, "paso": 0.5, "unidad": ""},
    "poblacion_por_casa": {"etiqueta": "Población (por casa)", "min": 1, "max": 8, "paso": 0.5, "unidad": "personas"},
    "ingreso_medio": {"etiqueta": "Ingreso medio de la zona", "min": 0, "max": 15, "paso": 0.5, "unidad": ""},
    "antiguedad": {"etiqueta": "Antigüedad de la casa", "min": 1, "max": 52, "paso": 1, "unidad": "años"},
    "latitud": {"etiqueta": "Latitud", "min": 32, "max": 42, "paso": 0.5, "unidad": ""},
    "longitud": {"etiqueta": "Longitud", "min": -124.5, "max": -114, "paso": 0.5, "unidad": ""},
}

variables = []
for c in base:
    meta = num_meta[c]
    variable = {
        "id": c,
        "etiqueta": meta["etiqueta"],
        "tipo": "numero",
        "media": float(medias[c]),
        "desviacion": float(desv[c]),
        "min": meta["min"],
        "max": meta["max"],
        "paso": meta["paso"],
        "unidad": meta["unidad"],
    }
    if c == "poblacion_por_casa":
        variable["transformacion"] = "log1p"
    variables.append(variable)

opciones = []
for nivel in oc:
    fid = f"mar_{nivel}"
    opciones.append({
        "valor": nivel,
        "etiqueta": etiquetas_mar[nivel],
        "id": fid,
        "media": float(medias[fid]),
        "desviacion": float(desv[fid]),
    })
variables.append({
    "id": "cercania_mar",
    "etiqueta": "Cercanía al mar",
    "tipo": "categoria",
    "opciones": opciones,
})

terminos = []
nombres = list(X.columns)
for fila_potencias, coef in zip(polinomio.powers_, modelo.coef_):
    factores = []
    for idx, potencia in enumerate(fila_potencias):
        if potencia > 0:
            factores.append([nombres[idx], int(potencia)])
    terminos.append({"factores": factores, "coeficiente": float(coef)})

modelo_json = {
    "modelo": "Regresion Lineal Multiple (polinomial grado 2)",
    "descripcion": "Predice el precio de una casa a partir de sus caracteristicas.",
    "target": {"nombre": "precio", "etiqueta": "Precio de la casa (USD)"},
    "transformacion": "log1p",
    "fecha": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    "datos": {
        "registros": int(len(X)),
        "entrenamiento": int(len(X_train)),
        "prueba": int(len(X_test)),
    },
    "metricas": {
        "r2_entrenamiento": float(r2_train),
        "r2_prueba": float(r2_test),
        "rmse_dolares": rmse_usd,
        "rmse_log": rmse_log,
    },
    "variables": variables,
    "terminos": terminos,
    "intercepto": float(modelo.intercept_),
    "unidades": "precio en dolares (USD)",
}

with open(RUTA_JSON, "w", encoding="utf-8") as f:
    json.dump(modelo_json, f, ensure_ascii=False, indent=2)

with open(RUTA_JS, "w", encoding="utf-8-sig") as f:
    f.write("window.MODELO = ")
    f.write(json.dumps(modelo_json, ensure_ascii=False, indent=2))
    f.write(";\n")

print("Archivos generados: modelo.json y modelo.js")
print(f"Registros usados: {len(X)}")
print(f"R2 entrenamiento: {r2_train:.4f}")
print(f"R2 prueba: {r2_test:.4f}")
print(f"RMSE prueba: ${rmse_usd:,.0f}")