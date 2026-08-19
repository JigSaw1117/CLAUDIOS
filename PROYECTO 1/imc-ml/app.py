import numpy as np
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

app = Flask(__name__)

CATEGORIAS = [
    {"max": 18.5, "nombre": "Bajo peso", "color": "#7a8aa0"},
    {"max": 25, "nombre": "Peso normal", "color": "#1F4E3D"},
    {"max": 30, "nombre": "Sobrepeso", "color": "#a3662b"},
    {"max": 999, "nombre": "Obesidad", "color": "#8a2b1f"},
]

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
pendiente = modelo.coef_[0]
intercepto = modelo.intercept_

print("Datos de entrenamiento creados (verificacion):")
for d in DATOS:
    f = rasgo(d["peso"], d["altura"])
    print(f"  peso {d['peso']:>3} kg  altura {d['altura']} m  ->  x={f:.3f}  IMC={f:.2f}")
print(f"Modelo: IMC = {pendiente:.4f} * (peso/altura^2) + {intercepto:.4f}")
print(f"R^2 = {r2:.6f}")


def categoria_imc(valor):
    for c in CATEGORIAS:
        if valor < c["max"]:
            return c
    return CATEGORIAS[-1]


def formatear(valor):
    redondeado = round(valor, 1)
    if redondeado.is_integer():
        return str(int(redondeado))
    return str(redondeado)


@app.route("/")
def index():
    resultado = None
    error = None
    peso_texto = request.args.get("peso", "").strip()
    altura_texto = request.args.get("altura", "").strip()

    if peso_texto and altura_texto:
        try:
            peso = float(peso_texto.replace(",", "."))
            altura = float(altura_texto.replace(",", "."))
            if peso <= 0 or altura <= 0:
                error = "Peso y altura deben ser mayores a cero."
            else:
                x = rasgo(peso, altura)
                imc = modelo.predict([[x]])[0]
                cat = categoria_imc(imc)
                resultado = {
                    "imc": formatear(imc),
                    "nombre": cat["nombre"],
                    "color": cat["color"],
                }
        except ValueError:
            error = "Usa numeros, ej. peso 70 y altura 1.75."

    return render_template(
        "index.html",
        peso_texto=peso_texto,
        altura_texto=altura_texto,
        resultado=resultado,
        error=error,
        pendiente=pendiente,
        intercepto=intercepto,
        r2=r2,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)