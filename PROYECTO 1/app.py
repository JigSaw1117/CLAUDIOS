import numpy as np
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

app = Flask(__name__)

MODOS = {
    "Celsius a Fahrenheit": {"simbolo": "\u00b0C", "extra": "\u00b0F", "convertir": lambda x: (x * 9 / 5) + 32},
    "Celsius a Kelvin": {"simbolo": "\u00b0C", "extra": "K", "convertir": lambda x: x + 273.15},
    "Fahrenheit a Celsius": {"simbolo": "\u00b0F", "extra": "\u00b0C", "convertir": lambda x: (x - 32) * 5 / 9},
    "Fahrenheit a Kelvin": {"simbolo": "\u00b0F", "extra": "K", "convertir": lambda x: (x - 32) * 5 / 9 + 273.15},
    "Kelvin a Celsius": {"simbolo": "K", "extra": "\u00b0C", "convertir": lambda x: x - 273.15},
    "Kelvin a Fahrenheit": {"simbolo": "K", "extra": "\u00b0F", "convertir": lambda x: x * 9 / 5 - 459.67},
}

DATOS_X = np.array([-5.0, 0.0, 10.0, 20.0, 30.0]).reshape(-1, 1)
DATOS_Y = np.array([23.0, 32.0, 50.0, 68.0, 86.0])


def entrenar_modelo():
    modelo = LinearRegression()
    modelo.fit(DATOS_X, DATOS_Y)
    r2 = r2_score(DATOS_Y, modelo.predict(DATOS_X))
    return modelo, r2


modelo, r2 = entrenar_modelo()


def formatear(valor):
    redondeado = round(valor, 2)
    if redondeado.is_integer():
        return str(int(redondeado))
    return str(redondeado)


@app.route("/")
def index():
    resultado = None
    error = None

    modo = request.args.get("modo", "Celsius a Fahrenheit")
    if modo not in MODOS:
        modo = "Celsius a Fahrenheit"

    valor_texto = request.args.get("valor", "").strip()
    if valor_texto:
        try:
            valor = float(valor_texto.replace(",", "."))
            simb = MODOS[modo]["simbolo"]
            extra = MODOS[modo]["extra"]
            conv = MODOS[modo]["convertir"]
            resultado = f"{formatear(valor)} {simb}  =  {formatear(conv(valor))} {extra}"
        except ValueError:
            error = "Usa un numero, ej. 25, -10.5 o 300."

    pred_x = request.args.get("pred_x", "").strip()
    prediccion = None
    if pred_x:
        try:
            x = float(pred_x.replace(",", "."))
            prediccion = f"{formatear(x)} \u00b0C  =  {formatear(modelo.predict([[x]])[0])} \u00b0F"
        except ValueError:
            error = "Usa un numero valido para la prediccion."

    filas = [
        {"c": int(c), "f": int(f)}
        for c, f in zip(DATOS_X.flatten(), DATOS_Y)
    ]
    pendiente = modelo.coef_[0]
    intercepto = modelo.intercept_

    return render_template(
        "index.html",
        modos=MODOS,
        modo_seleccionado=modo,
        valor_texto=valor_texto,
        resultado=resultado,
        error=error,
        filas=filas,
        pendiente=pendiente,
        intercepto=intercepto,
        r2=r2,
        pred_x=pred_x,
        prediccion=prediccion,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)