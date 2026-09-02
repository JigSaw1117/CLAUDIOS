# Caso 3 — Diabetes
# Informe 2 de 2: **Regresión Polinomial**

**Dataset:** `sklearn.datasets.load_diabetes` (Efron, Hastie, Johnstone y Tibshirani, 2004)
**Contexto:** predecir la progresión cuantitativa de la enfermedad un año después del
inicio, a partir de 10 variables basales.
**Herramientas:** Python 3.13, scikit-learn 1.9, pandas, NumPy, SciPy, Matplotlib, Seaborn.
**Reproducción:** `python train.py` (semilla fija `random_state=42`).

> Este informe cubre **exclusivamente la Regresión Polinomial**.
> La Regresión Lineal Múltiple y el desarrollo completo de la Fase A se documentan en
> `INFORME_1_REGRESION_MULTIPLE.md`.

---

## Resumen ejecutivo

| | |
|---|---|
| Metodología | Regresión Polinomial (grados 1 a 3, con y sin regularización) |
| Muestras | 442 (331 entrenamiento / 111 prueba) |
| Términos generados | 10 (grado 1) · **65** (grado 2) · **285** (grado 3) |
| Mejor variante | Polinomial grado 2 + Ridge (α = 25.12) |
| R² validación cruzada | **0.4611 ± 0.083** |
| Comparación con el lineal múltiple | +0.0005 → **estadísticamente indistinguible** |
| Sin regularizar, grado 2 | R² CV cae de 0.4606 a **0.2979** |
| Sin regularizar, grado 3 | R² CV = **−171.12** (colapso total) |
| **Conclusión** | **La expansión polinomial no aporta mejora. Se descarta.** |

---

# FASE A — Preprocesamiento (resumen)

La Fase A es común a ambas metodologías y se desarrolla completa en el Informe 1. Se
resume aquí lo estrictamente necesario para interpretar los resultados polinomiales.

| Aspecto | Resultado |
|---|---|
| Dimensiones | 442 × 10, todas `float64` |
| Valores nulos | 0 → sin imputación |
| Filas duplicadas | 0 |
| Outliers (IQR) | 2 a 9 por variable, ninguno en el target → se conservan |
| Asimetría del target | 0.441 → sin transformación logarítmica |
| Escalado | `StandardScaler` dentro del `Pipeline` |
| Partición | 331 entrenamiento / 111 prueba (75/25), `random_state=42` |

## Por qué la multicolinealidad importa especialmente aquí

Es el punto que condiciona toda esta fase. Los VIF detectados en la Fase A son:

| Variable | VIF | | Par | r |
|---|---|---|---|---|
| `s1` | **59.20** | | `s1` – `s2` | **+0.897** |
| `s2` | **39.19** | | `s3` – `s4` | −0.738 |
| `s3` | **15.40** | | `s2` – `s4` | +0.660 |
| `s5` | **10.08** | | `s4` – `s5` | +0.618 |
| `s4` | 8.89 | | resto | < 0.55 |

La multicolinealidad es **estructural**: `s4` = `s1`/`s3` por definición, y `s2` (LDL)
es un componente de `s1` (colesterol total).

*Figura: `figuras/a_correlacion.png`*

**La expansión polinomial agrava este problema de forma multiplicativa.** Si `s1` y `s2`
ya están correlacionadas a r = 0.897, entonces `s1²`, `s2²` y `s1·s2` estarán
correlacionadas *entre sí y con las originales* de forma aún más extrema. La expansión
no crea información nueva: replica la existente en formas redundantes.

Este es el motivo de fondo por el que los resultados de esta fase son los que son.

---

# FASE B — Regresión Polinomial

## B.1 Formulación del modelo

La regresión polinomial **no es un modelo no lineal**: es una regresión lineal aplicada
sobre un conjunto de variables ampliado con potencias y productos cruzados. Sigue
siendo **lineal en los parámetros**, que es lo que define el método.

Para grado 2 con 10 variables originales:

```
ŷ = β₀ + Σ βᵢxᵢ + Σ βᵢᵢxᵢ² + Σ βᵢⱼxᵢxⱼ
```

`PolynomialFeatures(degree=d, include_bias=False)` genera automáticamente esos términos:

| Grado | Lineales | Cuadrados | Productos cruzados | **Total** |
|---|---|---|---|---|
| 1 | 10 | — | — | **10** |
| 2 | 10 | 10 | 45 | **65** |
| 3 | 10 | 10 | 45 + 220 | **285** |

El `Pipeline` completo es:

```python
Pipeline([
    ("poly", PolynomialFeatures(degree=d, include_bias=False)),
    ("sc",   StandardScaler()),
    ("lr",   LinearRegression())      # o RidgeCV
])
```

El orden importa: la estandarización se aplica **después** de la expansión, porque los
términos cuadráticos y cruzados tienen escalas radicalmente distintas de las originales
(p. ej. `age·bmi` ≈ 1285 frente a `age` ≈ 50).

## B.2 La relación crítica: parámetros frente a muestras

Antes de ver los resultados, conviene fijar la magnitud del problema:

| Grado | Términos (p) | Muestras entren. (n) | Ratio n / p |
|---|---|---|---|
| 1 | 10 | 331 | **33.1** — holgado |
| 2 | 65 | 331 | **5.1** — ajustado |
| 3 | 285 | 331 | **1.16** — crítico |

Con grado 3 hay prácticamente **un parámetro por cada observación**. Un modelo así puede
reproducir casi exactamente los datos de entrenamiento sin haber aprendido nada
generalizable. Es la definición operativa del sobreajuste.

## B.3 Resultados

| Grado | Términos | Regularización | R² entren. | R² prueba | R² CV 5-fold | Brecha train−CV |
|---|---|---|---|---|---|---|
| 1 | 10 | — (OLS) | 0.5190 | 0.4849 | 0.4606 | 0.058 |
| 1 | 10 | Ridge α = 1.59 | 0.5185 | 0.4862 | 0.4597 | 0.059 |
| **2** | **65** | **— (OLS)** | **0.6048** | **0.4242** | **0.2979** | **0.307** |
| 2 | 65 | Ridge α = 25.12 | 0.5373 | 0.4911 | **0.4611** | 0.076 |
| **3** | **285** | **— (OLS)** | **0.8960** | **−8.0873** | **−171.12** | **172.01** |
| 3 | 285 | Ridge α = 199.53 | 0.5361 | 0.4942 | 0.4603 | 0.076 |

*Figura: `figuras/b_curva_validacion.png` — curva de validación por grado.*

*Figura: `figuras/b_brecha.png`*

## B.4 Análisis del sobreajuste

Esta tabla es el resultado más instructivo del trabajo. Conviene leerla por partes.

### Grado 2 sin regularizar: el sobreajuste aparece

El R² de entrenamiento **sube** de 0.5190 a **0.6048**. Visto de forma aislada, parece
una mejora de 8.6 puntos de R²: el modelo ajusta mejor los datos que ha visto.

Pero el R² de validación cruzada **se hunde** de 0.4606 a **0.2979** — una pérdida de
35 %. La brecha entre entrenamiento y validación pasa de 0.058 a **0.307**, multiplicada
por más de cinco.

**Diagnóstico:** el modelo está memorizando el ruido particular del set de
entrenamiento. Los 55 términos adicionales no capturan estructura real del fenómeno;
capturan las peculiaridades de estos 331 pacientes concretos.

### Grado 3 sin regularizar: el colapso

Con 285 términos y 331 muestras, el modelo alcanza un R² de entrenamiento de **0.8960**.
Aparenta explicar el 90 % de la varianza.

Fuera de la muestra el resultado es catastrófico:

- R² de prueba: **−8.0873**
- R² de validación cruzada: **−171.12 ± 167.49**
- RMSE de prueba: **224.17** (frente a 53.37 del modelo lineal)
- MAE de prueba: **151.10** (frente a 41.55)

**Un R² negativo significa que el modelo predice peor que responder siempre la media.**
Un −171 significa que lo hace de forma catastróficamente peor. El MAE de 151.10 puntos
sobre una escala de 25 a 346 hace las predicciones completamente inservibles.

> **Éste es el argumento central contra evaluar un modelo por su ajuste en
> entrenamiento.** El modelo con mejor R² de entrenamiento de todo el estudio (0.8960)
> es, con enorme diferencia, el peor modelo de todos.

### Con Ridge: el sobreajuste desaparece, pero no aparece ninguna ganancia

La penalización L2 encoge los coeficientes de los términos superfluos. El efecto es
inmediato y doble:

| Grado | R² CV sin Ridge | R² CV con Ridge | α elegido por CV |
|---|---|---|---|
| 1 | 0.4606 | 0.4597 | 1.59 |
| 2 | **0.2979** | **0.4611** | 25.12 |
| 3 | **−171.12** | **0.4603** | 199.53 |

Obsérvese que **α crece con el grado** (1.59 → 25.12 → 199.53): la validación cruzada
detecta automáticamente que a mayor número de términos hace falta más penalización.

Pero el resultado es revelador: los tres grados **convergen al mismo R² de validación
cruzada, ≈ 0.460**, que es exactamente el del modelo lineal múltiple. Ridge no está
extrayendo información adicional de los términos polinomiales — está **anulándolos**
hasta dejar el modelo funcionalmente equivalente al lineal.

## B.5 Comparación directa con la Regresión Lineal Múltiple

| | Lineal Múltiple | Polinomial gr. 2 + Ridge |
|---|---|---|
| Términos | **10** | 65 |
| R² entrenamiento | 0.5190 | 0.5373 |
| R² prueba | 0.4849 | 0.4911 |
| **R² validación cruzada** | **0.4606** | **0.4611** |
| Desviación entre particiones | ±0.085 | ±0.083 |
| Brecha train−CV | **0.058** | 0.076 |
| RMSE prueba | 53.37 | 53.05 |
| MAE prueba | 41.55 | 41.59 |
| Coeficientes interpretables | **Sí** | No |

La diferencia en validación cruzada es de **+0.0005** a favor del polinomial, frente a
una desviación entre particiones de **±0.083**. La supuesta ventaja es **166 veces
menor que el ruido de la propia medición**: los dos modelos son **estadísticamente
indistinguibles**.

### Decisión: se descarta el modelo polinomial

Por el **principio de parsimonia** (navaja de Occam), entre dos modelos de rendimiento
equivalente se elige el más simple:

- **10 términos frente a 65** — seis veces menos parámetros.
- **Coeficientes interpretables.** En el modelo lineal, βᵢ es el efecto de la variable
  *i*. En el polinomial, el efecto de `bmi` se reparte entre `bmi`, `bmi²` y sus nueve
  productos cruzados: no hay lectura clínica posible.
- **Menor brecha train−CV** (0.058 frente a 0.076): generaliza de forma más estable.
- **Menor riesgo operativo.** El modelo polinomial depende críticamente de que α esté
  bien calibrado; el lineal no depende de ningún hiperparámetro.

## B.6 Interpretación de fondo

Que la expansión polinomial no aporte nada **no es un fracaso del experimento: es un
resultado con contenido**.

Significa que la relación entre las variables basales y la progresión de la enfermedad
es **esencialmente lineal en el rango observado**. No existe curvatura relevante ni
interacciones entre variables que el modelo lineal esté dejando escapar.

Esto es **coherente con el diagnóstico de residuos** del Informe 1 (sección B.4): los
residuos del modelo lineal se distribuyen de forma aproximadamente normal y sin patrón
estructural frente a las predicciones. Si hubiera curvatura sin capturar, los residuos
la habrían mostrado como un patrón sistemático, y la expansión polinomial la habría
aprovechado. Ni lo uno ni lo otro ocurre.

Dicho de otro modo: los dos análisis, hechos por caminos independientes, llegan a la
misma conclusión. **El techo de ~0.46 en validación cruzada no es una limitación de la
forma funcional del modelo, sino del contenido informativo de los datos** — 442
pacientes y 10 variables basales, sin información sobre genética, tratamiento,
adherencia, dieta ni comorbilidades.

Aumentar la complejidad del modelo no puede resolver una carencia de información.

---

# FASE C — Aplicativo web (modo Regresión Polinomial)

El aplicativo `index.html` incorpora **ambos modelos** con un selector, de modo que la
comparativa de la Fase B queda disponible de forma interactiva.

## Serialización del modelo polinomial

Exportar un modelo polinomial a JavaScript exige más que los coeficientes: hay que
reproducir exactamente la expansión de `PolynomialFeatures`. Para ello `train.py`
exporta la matriz de exponentes `PolynomialFeatures.powers_`, de 65 × 10:

```javascript
/** Expande el vector base según los exponentes exportados por PolynomialFeatures. */
function expandir(x, powers) {
  return powers.map(p => {
    let t = 1;
    for (let j = 0; j < p.length; j++) {
      if (p[j] === 1) t *= x[j];
      else if (p[j] > 1) t *= Math.pow(x[j], p[j]);
    }
    return t;
  });
}
```

Cada fila de `powers` indica el exponente de cada variable en ese término. Por ejemplo,
la fila `[1,0,1,0,0,0,0,0,0,0]` corresponde al término `age·bmi`, y
`[2,0,0,0,0,0,0,0,0,0]` a `age²`. Con esto el aplicativo reconstruye los 65 términos en
el mismo orden que scikit-learn, aplica μᵢ y σᵢ, y calcula el producto punto.

## Funcionalidades

- **Selector de modelo** entre Lineal Múltiple y Polinomial grado 2, que actualiza
  predicción, métricas y desglose.
- **Cálculo paso a paso** que muestra la construcción de los términos con un ejemplo de
  cada tipo (lineal, cuadrático y producto cruzado), la estandarización y la suma
  ponderada.
- **Tabla de detalle** con los 40 términos de mayor aporte de los 65.
- **Tabla comparativa** de la Fase B con ambos modelos.

## Verificación de paridad con scikit-learn

| Caso de prueba | Python (65 términos) | JavaScript |
|---|---|---|
| Medianas del dataset | 151.380316 | 151.38031624715595 |
| Primer registro del set de prueba | 138.116160 | 138.1161603440222 |
| Perfil de riesgo alto | 344.651909 | 344.6519087280618 |

Coincidencia con precisión de punto flotante, incluida la expansión completa de los
65 términos.

## Ejecución y despliegue

```bash
python -m http.server 8000
```

Para publicarlo basta con subir `index.html` y `model.js` a cualquier hosting estático.

---

# Conclusiones — Regresión Polinomial

1. **La regresión polinomial sigue siendo lineal en los parámetros.** Se implementa
   como una regresión lineal sobre variables ampliadas con potencias y productos
   cruzados mediante `PolynomialFeatures`.

2. **Sin regularización, el sobreajuste es severo y creciente con el grado.** Grado 2
   hunde el R² de validación cruzada de 0.4606 a 0.2979; grado 3 lo lleva a **−171.12**,
   es decir, a predecir mucho peor que la media, pese a alcanzar un R² de entrenamiento
   de 0.8960.

3. **El caso de grado 3 demuestra por qué no debe evaluarse un modelo por su ajuste en
   entrenamiento.** El modelo con el mejor R² de entrenamiento de todo el estudio es el
   peor de todos fuera de la muestra.

4. **Ridge controla el sobreajuste eficazmente**, y α crece automáticamente con el grado
   (1.59 → 25.12 → 199.53). Pero los tres grados convergen al mismo R² de validación
   cruzada (≈ 0.460): la penalización no extrae información de los términos
   polinomiales, los neutraliza.

5. **El modelo polinomial no supera al lineal múltiple.** La diferencia de +0.0005 en
   validación cruzada es 166 veces menor que la desviación entre particiones (±0.083).
   **Se descarta por parsimonia**: 65 términos no interpretables frente a 10
   interpretables, con idéntico rendimiento.

6. **La relación es esencialmente lineal en el rango observado.** Esta conclusión
   coincide con el diagnóstico de residuos del Informe 1, obtenido por una vía
   independiente. El límite de ~0.46 proviene del contenido informativo de los datos,
   no de la forma funcional del modelo.

7. **La multicolinealidad de la Fase A explica el resultado.** La expansión polinomial
   sobre variables ya estructuralmente colineales (`s4` = `s1`/`s3`, r(`s1`,`s2`) =
   0.897) genera términos redundantes, no información nueva.

---

# Referencias

1. Efron, B., Hastie, T., Johnstone, I. y Tibshirani, R. (2004). *Least Angle
   Regression*. **Annals of Statistics**, 32(2), 407–499.
   https://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf
2. Pedregosa, F. *et al.* (2011). *Scikit-learn: Machine Learning in Python*.
   **Journal of Machine Learning Research**, 12, 2825–2830.
3. Documentación de scikit-learn — `sklearn.preprocessing.PolynomialFeatures` y
   `sklearn.linear_model.RidgeCV`.
   https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression
4. James, G., Witten, D., Hastie, T. y Tibshirani, R. (2021). *An Introduction to
   Statistical Learning*, 2.ª ed. Springer. Capítulo 6: Métodos de Regularización;
   Capítulo 7: Más allá de la linealidad.
5. Hastie, T., Tibshirani, R. y Friedman, J. (2009). *The Elements of Statistical
   Learning*, 2.ª ed. Springer. (Compromiso sesgo-varianza).
6. Fuente original de los datos:
   https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
