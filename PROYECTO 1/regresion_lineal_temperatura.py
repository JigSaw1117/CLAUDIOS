import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

celsius = np.array([-5.0, 0.0, 10.0, 20.0, 30.0]).reshape(-1, 1)
fahrenheit = np.array([23.0, 32.0, 50.0, 68.0, 86.0])

modelo = LinearRegression()
modelo.fit(celsius, fahrenheit)

pendiente = modelo.coef_[0]
intercepto = modelo.intercept_

print("Datos de entrenamiento:")
for c, f in zip(celsius.flatten(), fahrenheit):
    print(f"  {c:>5} C -> {f:>5} F")

print("\nModelo de regresion lineal simple:")
print(f"  Formula aprendida:  F = {pendiente:.6f} * C + {intercepto:.6f}")

predicciones = modelo.predict(celsius)
r2 = r2_score(fahrenheit, predicciones)
print(f"  R^2 (precision del modelo): {r2:.10f}")

print("\nPredicciones del modelo sobre los datos de entrada:")
for c, real, pred in zip(celsius.flatten(), fahrenheit, predicciones):
    print(f"  {c:>5} C -> predicho {pred:>6.2f} F  | real {real:>5} F  | error {pred - real:+.2f}")

nuevos = np.array([100.0, -40.0]).reshape(-1, 1)
print("\nPredicciones para nuevos valores:")
for n in nuevos.flatten():
    print(f"  {n:>5} C -> {modelo.predict([[n]])[0]:.2f} F")