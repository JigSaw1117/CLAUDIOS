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
    (52, 1.60), (70, 1.75), (64, 1.62), (88, 1.70),
    (45, 1.52), (97, 1.72), (110, 1.90), (60, 1.63),
]

X = np.array([[p, a] for p, a in DATOS])
y = np.array([p / (a * a) for p, a in DATOS])

modelo = LinearRegression()
modelo.fit(X, y)
r2 = r2_score(y, modelo.predict(X))
b_peso = float(modelo.coef_[0])
b_altura = float(modelo.coef_[1])
intercepto = float(modelo.intercept_)

print("Regresion MULTIPLE: IMC = B0 + B1*peso + B2*altura")
print(f"  B0 (intercepto) = {intercepto:.6f}")
print(f"  B1 (peso)       = {b_peso:.6f}")
print(f"  B2 (altura)     = {b_altura:.6f}")
print(f"  R2 = {r2:.6f} | N = {len(DATOS)}")


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
                imc = modelo.predict([[peso, altura]])[0]
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
        b_peso=b_peso,
        b_altura=b_altura,
        intercepto=intercepto,
        r2=r2,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)