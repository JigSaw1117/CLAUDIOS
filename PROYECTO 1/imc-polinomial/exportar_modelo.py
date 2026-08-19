import json
import numpy as np
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

DATOS = [
    (52, 1.60), (70, 1.75), (64, 1.62), (88, 1.70),
    (45, 1.52), (97, 1.72), (110, 1.90), (60, 1.63),
]

X = np.array([[p, a] for p, a in DATOS])
y = np.array([p / (a * a) for p, a in DATOS])

GRAD = 2
transformador = PolynomialFeatures(degree=GRAD, include_bias=False)
modelo = make_pipeline(transformador, LinearRegression())
modelo.fit(X, y)

r2 = r2_score(y, modelo.predict(X))
features = transformador.get_feature_names_out(["peso", "altura"]).tolist()
coefs = modelo.named_steps["linearregression"].coef_.tolist()
intercepto = float(modelo.named_steps["linearregression"].intercept_)

dump(modelo, "modelo_imc_pol.pkl")

for i, c in enumerate(coefs):
    coefs[i] = float(c)

artefacto = {
    "modelo": "PolynomialFeatures + LinearRegression",
    "grado": GRAD,
    "variables": ["peso", "altura"],
    "features": features,
    "coef": coefs,
    "intercepto": intercepto,
    "r2": float(r2),
    "n": len(DATOS),
}

with open("model.json", "w", encoding="utf-8") as f:
    json.dump(artefacto, f, ensure_ascii=False, indent=2)

print("Artefacto .pkl  -> modelo_imc_pol.pkl")
print("Artefacto .json -> model.json")
print("R2 = %.6f" % r2)
for nombre, c in zip(features, coefs):
    print("  %s = %.6f" % (nombre, c))
print("  intercepto = %.6f" % intercepto)