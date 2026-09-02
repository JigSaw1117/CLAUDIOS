# Caso 3 — Diabetes
# Informe 1 de 2: **Regresión Lineal Múltiple**

**Dataset:** `sklearn.datasets.load_diabetes` (Efron, Hastie, Johnstone y Tibshirani, 2004)
**Contexto:** predecir la progresión cuantitativa de la enfermedad un año después del
inicio, a partir de 10 variables basales.
**Herramientas:** Python 3.13, scikit-learn 1.9, pandas, NumPy, SciPy, Matplotlib, Seaborn.
**Reproducción:** `python train.py` (semilla fija `random_state=42`).

> Este informe cubre **exclusivamente la Regresión Lineal Múltiple**.
> La Regresión Polinomial se documenta en `INFORME_2_REGRESION_POLINOMIAL.md`.
> La Fase A es común a ambas metodologías y se desarrolla completa aquí.

---

## Resumen ejecutivo

| | |
|---|---|
| Metodología | Regresión Lineal Múltiple (OLS, Ridge, Lasso) |
| Muestras | 442 (331 entrenamiento / 111 prueba) |
| Variables | 10 predictoras, todas numéricas |
| Modelo final | **OLS con las 10 variables** |
| R² entrenamiento | 0.5190 |
| R² prueba | **0.4849** |
| R² validación cruzada 5-fold | **0.4606 ± 0.085** |
| RMSE / MAE prueba | 53.37 / 41.55 puntos |
| Brecha entrenamiento − CV | 0.0584 → **sin sobreajuste** |
| Predictores significativos | `bmi`, `s5`, `bp`, `sex`, `s1` |

---

# FASE A — Análisis exploratorio y preprocesamiento

## A.0 Carga del dataset y decisión sobre la escala

```python
X, y = load_diabetes(return_X_y=True, as_frame=True, scaled=False)
```

`load_diabetes()` devuelve por defecto (`scaled=True`) los datos **ya centrados y
escalados** por la desviación estándar × √n, con valores adimensionales en el rango
−0.14 a 0.20. Se usó **`scaled=False`** de forma deliberada, por dos razones:

1. La Fase A exige aplicar `StandardScaler`/`MinMaxScaler`; no tiene sentido
   estandarizar datos ya estandarizados, y el procedimiento no sería demostrable.
2. La Fase C exige que el usuario ingrese parámetros. Un usuario introduce
   "IMC = 26.4", no "IMC = 0.0507".

## A.1 Tipos de datos, valores nulos y duplicados

442 filas × 10 columnas. **Todas de tipo `float64`.**

| Variable | Descripción | Unidad | Media | Desv. | Mín | Máx |
|---|---|---|---|---|---|---|
| `age` | Edad | años | 48.52 | 13.11 | 19.0 | 79.0 |
| `sex` | Sexo (categórica binaria) | 0 / 1 | 0.47 | 0.50 | 0 | 1 |
| `bmi` | Índice de masa corporal | kg/m² | 26.38 | 4.42 | 18.0 | 42.2 |
| `bp` | Presión arterial media | mm Hg | 94.65 | 13.83 | 62.0 | 133.0 |
| `s1` | Colesterol total (tc) | mg/dL | 189.14 | 34.61 | 97.0 | 301.0 |
| `s2` | Lipoproteínas LDL (ldl) | mg/dL | 115.44 | 30.41 | 41.6 | 242.4 |
| `s3` | Lipoproteínas HDL (hdl) | mg/dL | 49.79 | 12.93 | 22.0 | 99.0 |
| `s4` | Razón colesterol total / HDL (tch) | ratio | 4.07 | 1.29 | 2.0 | 9.09 |
| `s5` | Triglicéridos séricos, *posiblemente* en log (ltg) | log | 4.64 | 0.52 | 3.26 | 6.11 |
| `s6` | Glucosa en sangre (glu) | mg/dL | 91.26 | 11.50 | 58.0 | 124.0 |

> **Nota sobre `s1`–`s6`:** los nombres provienen del `.DESCR` oficial. La propia
> documentación de scikit-learn advierte que su significado exacto *"puede no ser
> claro, especialmente para `ltg`"*, porque el dataset original no lo documenta de
> forma explícita. Se reporta con esa reserva.

**Variable objetivo:** medida cuantitativa de progresión de la enfermedad a un año.
Continua, rango 25 – 346, media 152.13, **asimetría 0.441**.

### Resultado de la limpieza

| Verificación | Resultado | Acción |
|---|---|---|
| Valores nulos | **0** | No se requiere imputación |
| Filas duplicadas | **0** | No se requiere deduplicación |
| Asimetría del target | 0.441 (aceptable) | **No** se aplica transformación logarítmica |

La ausencia de nulos se verificó explícitamente (`.isna().sum()`, `.duplicated().sum()`)
y se documenta para dejar constancia de que la comprobación se realizó.

### Recodificación de la variable categórica

`sex` es la única variable categórica y viene codificada como 1/2. Se recodificó a
**0/1** para que su coeficiente se interprete como el efecto de una categoría respecto
a la otra. No debe leerse como una variable continua.

## A.2 Tratamiento de outliers

Detección por el método del rango intercuartílico (IQR, k = 1.5):

| Variable | Outliers | % |
|---|---|---|
| `s6` | 9 | 2.0 % |
| `s1` | 8 | 1.8 % |
| `s2` | 7 | 1.6 % |
| `s3` | 7 | 1.6 % |
| `s5` | 4 | 0.9 % |
| `bmi` | 3 | 0.7 % |
| `s4` | 2 | 0.5 % |
| **target** | **0** | — |

*Figuras: `figuras/a3_boxplots.png`, `figuras/a3_target.png`*

### Decisión: se conservan. Justificación cuantitativa

Los valores atípicos son **clínicamente plausibles** — un colesterol total de
301 mg/dL es alto, pero perfectamente real en una población diabética — y con solo
442 muestras cada observación es valiosa. La variable objetivo no presenta outliers.

Para no dejar la decisión en lo cualitativo, se entrenó un modelo de control con los
datos **winsorizados a los percentiles 1–99**:

| Tratamiento | R² prueba |
|---|---|
| Sin tratar outliers | 0.4849 |
| Winsorizado 1 % – 99 % | 0.4865 |
| **Diferencia** | **+0.0016** |

La mejora es despreciable frente a la desviación de la validación cruzada (±0.085).
**Se confirma la decisión de conservarlos.**

## A.3 Análisis de multicolinealidad

### Matriz de correlación de Pearson

*Figura: `figuras/a4_correlacion.png`*

Pares con |r| > 0.5:

| Par | r |
|---|---|
| `s1` – `s2` | **+0.897** |
| `s3` – `s4` | −0.738 |
| `s2` – `s4` | +0.660 |
| `s4` – `s5` | +0.618 |
| `s1` – `s4` | +0.542 |
| `s1` – `s5` | +0.516 |

### Factor de Inflación de la Varianza (VIF)

Calculado como VIF*ⱼ* = 1 / (1 − R²*ⱼ*), donde R²*ⱼ* es el ajuste de la variable *j*
regresada contra todas las demás.

*Figura: `figuras/a4_vif.png`*

| Variable | VIF | Diagnóstico |
|---|---|---|
| `s1` | **59.20** | Severa |
| `s2` | **39.19** | Severa |
| `s3` | **15.40** | Severa |
| `s5` | **10.08** | Severa |
| `s4` | 8.89 | Moderada |
| `bmi` | 1.51 | Aceptable |
| `s6` | 1.48 | Aceptable |
| `bp` | 1.46 | Aceptable |
| `sex` | 1.28 | Aceptable |
| `age` | 1.22 | Aceptable |

### Interpretación: la multicolinealidad es estructural, no casual

No se trata de una correlación fortuita, sino de una **dependencia por definición**:

- `s4` = colesterol total / HDL = **`s1` / `s3`**. Es literalmente un cociente de otras
  dos variables del conjunto.
- `s2` (LDL) es un **componente** de `s1` (colesterol total), que agrupa LDL + HDL +
  otras fracciones.

Las cinco variables lipídicas (`s1`–`s5`) describen el mismo perfil desde ángulos
solapados. **Consecuencia práctica:** sus coeficientes individuales no son
interpretables por separado, porque el modelo no puede atribuir el efecto a una u otra.
Esto se confirma numéricamente en la sección B.2 de este informe.

Las variables clínicamente independientes (`age`, `sex`, `bmi`, `bp`, `s6`) tienen VIF
por debajo de 1.6, es decir, sin problema alguno.

## A.4 Estandarización

Se aplicó `StandardScaler` (media 0, desviación 1) **dentro de un `Pipeline` de
scikit-learn**:

```python
Pipeline([("sc", StandardScaler()), ("lr", LinearRegression())])
```

Esto es esencial y no meramente cosmético: al estar dentro del `Pipeline`, la media y
la desviación se recalculan **con los datos de entrenamiento de cada partición** en la
validación cruzada. Si se estandarizara todo el conjunto antes de partirlo, la
información del set de prueba se filtraría al entrenamiento (*data leakage*) y las
métricas quedarían infladas.

Se eligió `StandardScaler` sobre `MinMaxScaler` porque la regresión con regularización
(Ridge/Lasso) penaliza la magnitud de los coeficientes, y esa penalización solo es
equitativa si todas las variables comparten escala de varianza.

## A.5 Partición de datos

| | Muestras | Proporción |
|---|---|---|
| Entrenamiento | 331 | 75 % |
| Prueba | 111 | 25 % |

`train_test_split(X, y, test_size=0.25, random_state=42)`.

> **Advertencia metodológica.** Con solo 111 muestras de prueba, el R² de test es una
> medida ruidosa: reordenar la partición puede moverlo varias centésimas. Por eso
> **todas las comparaciones se hacen sobre la validación cruzada 5-fold** del set de
> entrenamiento, reportando además su desviación estándar. El R² de prueba se incluye
> únicamente como comprobación final independiente.

---

# FASE B — Regresión Lineal Múltiple

## B.1 Formulación del modelo

La regresión lineal múltiple modela la variable objetivo como combinación lineal de
los *p* predictores:

```
ŷ = β₀ + β₁z₁ + β₂z₂ + … + β₁₀z₁₀        con  zᵢ = (xᵢ − μᵢ) / σᵢ
```

Los coeficientes se estiman por **mínimos cuadrados ordinarios (OLS)**, minimizando la
suma de residuos al cuadrado:

```
J(β) = Σⱼ ( yⱼ − ŷⱼ )²
```

Las variantes regularizadas añaden una penalización sobre la magnitud de los
coeficientes:

```
Ridge (L2):  J(β) = Σ ( yⱼ − ŷⱼ )² + α · Σ βᵢ²
Lasso (L1):  J(β) = Σ ( yⱼ − ŷⱼ )² + α · Σ |βᵢ|
```

En ambos casos α se seleccionó por validación cruzada 5-fold (`RidgeCV`, `LassoCV`).

## B.2 Variantes entrenadas y resultados

Cuatro variantes, todas con `StandardScaler` dentro del `Pipeline`:

| Modelo | Términos | R² entren. | R² prueba | R² CV 5-fold | Brecha train−CV |
|---|---|---|---|---|---|
| **OLS — 10 variables** | 10 | 0.5190 | 0.4849 | **0.4606 ± 0.085** | 0.0584 |
| OLS — sin `s1`,`s2` (corte por VIF) | 8 | 0.5078 | **0.4964** | 0.4504 ± 0.096 | 0.0574 |
| Ridge (L2, α = 1.585) | 10 | 0.5185 | 0.4862 | 0.4597 ± 0.084 | 0.0587 |
| Lasso (L1, α = 0.114) | 10 | 0.5185 | 0.4868 | 0.4577 ± 0.091 | 0.0608 |

### Lecturas

**No hay sobreajuste.** La brecha entre entrenamiento y validación cruzada es de
~0.058 en las cuatro variantes: el modelo generaliza correctamente.

**La regularización casi no cambia nada.** Con 331 muestras y solo 10 variables, el
modelo no tiene capacidad suficiente para memorizar los datos, de modo que Ridge y
Lasso apenas mueven las métricas (±0.003 en CV). Lasso anula una única variable,
`s3` — coherente con que `s3` participa en la definición de `s4` y aporta poca
información propia.

> Este resultado es relevante para el Informe 2: la regularización aquí es
> **innecesaria**, pero se vuelve **imprescindible** al aplicarla sobre la expansión
> polinomial, donde el número de parámetros sí desborda a las muestras disponibles.

**Eliminar `s1` y `s2` mejora el R² de prueba (0.4964)** pero empeora el de validación
cruzada (0.4504). Como la CV es el criterio más fiable con 111 muestras de prueba, no
se adopta como modelo final; sin embargo confirma que esas dos variables aportan poca
información única, tal como anticipaba su VIF.

**Modelo final: OLS con las 10 variables**, por tener el mejor R² de validación cruzada
(0.4606) y la interpretación más directa.

## B.3 Significancia estadística de los coeficientes

Prueba *t* sobre los coeficientes del modelo OLS con las 10 variables estandarizadas
(gl = 331 − 11 = 320):

| Variable | Coeficiente | Error estándar | t | p-valor | Signif. |
|---|---|---|---|---|---|
| intercepto | 154.3444 | 3.0142 | 51.21 | < 0.0001 | *** |
| `age` | 2.2151 | 3.3234 | 0.67 | 0.5056 | |
| `sex` | −11.5145 | 3.3670 | −3.42 | 0.0007 | *** |
| `bmi` | **25.0770** | 3.7263 | 6.73 | < 0.0001 | *** |
| `bp` | **18.2493** | 3.5898 | 5.08 | < 0.0001 | *** |
| `s1` | −44.1446 | **22.2102** | −1.99 | 0.0477 | * |
| `s2` | 24.5139 | **17.9831** | 1.36 | 0.1738 | |
| `s3` | 5.4976 | **11.2118** | 0.49 | 0.6242 | |
| `s4` | 13.0068 | 9.2005 | 1.41 | 0.1584 | |
| `s5` | **33.3798** | 9.4361 | 3.54 | 0.0005 | *** |
| `s6` | 1.2480 | 3.6435 | 0.34 | 0.7322 | |

*Significancia: \*\*\* p < 0.001, \*\* p < 0.01, \* p < 0.05*

**Variables significativas al 5 %: `sex`, `bmi`, `bp`, `s1`, `s5`.**

### Confirmación numérica del diagnóstico de la Fase A

El patrón es inequívoco. Las variables **no significativas** (`s2`, `s3`, `s4`, `s6`,
`age`) incluyen justamente las de **VIF más alto**, y sus errores estándar lo delatan:

| Variable | VIF | Error estándar |
|---|---|---|
| `s1` | 59.20 | **22.21** |
| `s2` | 39.19 | **17.98** |
| `s3` | 15.40 | **11.21** |
| `age` | 1.22 | 3.32 |
| `sex` | 1.28 | 3.37 |
| `bmi` | 1.51 | 3.73 |

Los errores estándar de las variables colineales son entre **5 y 7 veces mayores**.
Eso es exactamente lo que produce la multicolinealidad: **infla la varianza de los
estimadores**. El modelo detecta que el perfil lipídico influye, pero no puede repartir
el crédito entre variables que se contienen unas a otras por definición.

**Implicación práctica:** los coeficientes lipídicos individuales **no deben
interpretarse clínicamente**. El modelo sigue siendo válido para *predecir* — la
multicolinealidad no sesga las predicciones — pero no para *explicar* el efecto
aislado de `s1` o `s2`.

Los predictores clínicamente sólidos y estadísticamente limpios son **`bmi`** (+25.08),
**`s5`** (+33.38), **`bp`** (+18.25) y **`sex`** (−11.51).

*Figura: `figuras/b_coeficientes.png`*

## B.4 Diagnóstico de residuos

*Figura: `figuras/b_residuos.png` — residuos vs. predicción, histograma y gráfico Q-Q.*
*Figura: `figuras/b_real_vs_predicho.png`*

Los residuos presentan distribución **aproximadamente normal** y **sin patrón
estructural** frente a las predicciones. Esto valida los supuestos del modelo lineal e
indica que el error restante es **ruido aleatorio, no sesgo sistemático sin capturar**.

Es un resultado importante: si los residuos mostraran curvatura, habría margen para un
modelo no lineal. No la muestran — lo que anticipa el resultado del Informe 2.

## B.5 Sobre la precisión alcanzada

Un R² de 0.485 puede parecer bajo. **No lo es para este problema**, y conviene
argumentarlo con precisión:

1. **Límite intrínseco de los datos.** Solo 442 pacientes y 10 variables basales. La
   progresión de una enfermedad crónica depende de factores no medidos aquí: carga
   genética, tratamiento recibido, adherencia terapéutica, dieta, actividad física,
   comorbilidades.
2. **Referencia de la literatura.** El artículo que introdujo este dataset —Efron,
   Hastie, Johnstone y Tibshirani (2004), *"Least Angle Regression"*— trabaja en este
   mismo rango de ajuste. No es un resultado atípico.
3. **El diagnóstico de residuos es correcto**, como se documentó en B.4: el error
   restante es ruido, no estructura sin modelar.
4. **Un R² alto sería sospechoso.** Como se demuestra en el Informe 2, el modelo que
   alcanza R² = 0.896 en entrenamiento resulta ser el **peor de todos** al evaluarlo
   fuera de la muestra.

El modelo explica cerca de la mitad de la varianza de un fenómeno biológico complejo,
con un error medio de ±41.5 puntos sobre una escala de 25 a 346.

---

# FASE C — Aplicativo web (modo Regresión Lineal Múltiple)

`index.html` + `model.js` — HTML y JavaScript puro, sin dependencias externas ni
proceso de compilación.

`train.py` serializa el modelo en `model.js`: los parámetros de estandarización
(μᵢ, σᵢ), los coeficientes βᵢ y el intercepto β₀. El aplicativo reconstruye la
predicción con la misma fórmula de la sección B.1.

### Funcionalidades relevantes a este informe

- **Entrada dinámica** de las 10 variables basales mediante deslizadores acotados al
  rango real observado, sincronizados con campos numéricos. Predicción en tiempo real.
- **Cálculo paso a paso**: estandarización de cada variable, producto por su
  coeficiente, suma y adición del intercepto, con los números reales de la predicción.
- **Tabla de detalle** por variable con xᵢ, μᵢ, σᵢ, zᵢ, βᵢ y βᵢ·zᵢ.
- **Tabla de VIF** de la Fase A.
- **Aviso** de que se trata de un ejercicio académico, no de una herramienta de
  diagnóstico clínico.

### Verificación de paridad con scikit-learn

| Caso de prueba | Python (scikit-learn) | JavaScript |
|---|---|---|
| Medianas del dataset | 155.862491 | 155.8624910825692 |
| Primer registro del set de prueba | 137.949089 | 137.9490887781244 |
| Perfil de riesgo alto | 300.618775 | 300.6187747290852 |

Coincidencia con precisión de punto flotante.

### Ejecución y despliegue

El archivo carga `model.js` mediante `<script src>`, que algunos navegadores bloquean
bajo el protocolo `file://`. Servirlo por HTTP:

```bash
python -m http.server 8000
```

Para publicarlo basta con subir `index.html` y `model.js` a cualquier hosting estático
(Netlify, GitHub Pages, Render). No requiere build ni servidor de aplicación.

---

# Conclusiones — Regresión Lineal Múltiple

1. **El dataset es adecuado** para la metodología: variable objetivo continua, 10
   predictores numéricos, sin nulos ni duplicados. Requiere `scaled=False` para poder
   aplicar el preprocesamiento exigido y construir una interfaz usable.

2. **La multicolinealidad es el hallazgo dominante** y es estructural: las variables
   lipídicas se contienen unas a otras por definición (`s4` = `s1`/`s3`, `s2` ⊂ `s1`),
   con VIF de hasta 59.2. Su efecto se manifiesta como errores estándar entre 5 y 7
   veces mayores en los coeficientes afectados, que en consecuencia **no son
   interpretables individualmente**.

3. **La regularización no mejora el ajuste.** Con 10 variables y 331 muestras el modelo
   no tiene margen para sobreajustar, de modo que Ridge y Lasso no cambian las métricas.

4. **El modelo final es OLS con las 10 variables**: R² de prueba 0.4849, R² de
   validación cruzada 0.4606 ± 0.085, RMSE 53.37 y MAE 41.55. Los predictores más
   sólidos son `bmi`, `s5`, `bp` y `sex`.

5. **Los residuos validan el supuesto de linealidad**: distribución normal y sin patrón
   estructural. El error restante es ruido, no sesgo sistemático.

6. **La validación cruzada fue determinante.** Escogiendo por R² de prueba se habría
   seleccionado el modelo sin `s1`,`s2` (0.4964); la CV muestra que esa diferencia no
   supera el ruido de muestreo.

---

# Referencias

1. Efron, B., Hastie, T., Johnstone, I. y Tibshirani, R. (2004). *Least Angle
   Regression*. **Annals of Statistics**, 32(2), 407–499.
   https://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf
2. Pedregosa, F. *et al.* (2011). *Scikit-learn: Machine Learning in Python*.
   **Journal of Machine Learning Research**, 12, 2825–2830.
3. Documentación de scikit-learn — `sklearn.datasets.load_diabetes`.
   https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
4. James, G., Witten, D., Hastie, T. y Tibshirani, R. (2021). *An Introduction to
   Statistical Learning*, 2.ª ed. Springer. Capítulo 3: Regresión Lineal.
5. Montgomery, D. C., Peck, E. A. y Vining, G. G. (2021). *Introduction to Linear
   Regression Analysis*, 6.ª ed. Wiley. (Diagnóstico de multicolinealidad y VIF).
6. Fuente original de los datos:
   https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
