import json
import numpy as np
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

DATOS = [
    {"peso": 52, "altura": 1.60},
    {"peso": 70, "altura": 1.75},
    {"peso": 64, "altura": 1.62},
    {"peso": 88, "altura": 1.70},
    {"peso": 45, "altura": 1.52},
    {"peso": 97, "altura": 1.72},
    {"peso": 110, "altura": 1.90},
    {"peso": 60, "altura": 1.63},
]


def rasgo(peso, altura):
    return peso / (altura * altura)


X = np.array([[rasgo(d["peso"], d["altura"])] for d in DATOS])
y = np.array([rasgo(d["peso"], d["altura"]) for d in DATOS])

modelo = LinearRegression()
modelo.fit(X, y)
r2 = r2_score(y, modelo.predict(X))

dump(modelo, "modelo_imc.pkl")

artefacto = {
    "modelo": "LinearRegression",
    "rasgo": "peso / (altura^2)",
    "pendiente": float(modelo.coef_[0]),
    "intercepto": float(modelo.intercept_),
    "r2": float(r2),
    "n_muestras": len(DATOS),
}
with open("model.json", "w", encoding="utf-8") as f:
    json.dump(artefacto, f, ensure_ascii=False, indent=2)

print("Artefacto .pkl  -> modelo_imc.pkl")
print("Artefacto .json -> model.json")
print(json.dumps(artefacto, indent=2))