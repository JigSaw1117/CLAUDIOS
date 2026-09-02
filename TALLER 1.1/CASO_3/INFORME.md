# Caso 3 — Diabetes
# Informe Técnico Consolidado

**Dataset:** `sklearn.datasets.load_diabetes` (Efron, Hastie, Johnstone y Tibshirani, 2004)
**Contexto:** predecir la progresión cuantitativa de la enfermedad un año después del
inicio, a partir de 10 variables basales.
**Reproducción:** `python train.py` (semilla fija `random_state=42`).

---

## Estructura de la documentación

Las dos metodologías exigidas se documentan **por separado**, cada una en su propio
informe autocontenido:

| Documento | Contenido |
|---|---|
| **[Informe 1 — Regresión Lineal Múltiple](INFORME_1_REGRESION_MULTIPLE.md)** | Fase A completa (EDA y preprocesamiento) + Fase B múltiple (OLS, Ridge, Lasso, significancia, residuos) + Fase C |
| **[Informe 2 — Regresión Polinomial](INFORME_2_REGRESION_POLINOMIAL.md)** | Fase A resumida + Fase B polinomial (grados 1–3, análisis de sobreajuste, comparativa) + Fase C |
| Este documento | Síntesis, comparativa global y conclusiones conjuntas |

La **Fase A es común** a ambas metodologías y se desarrolla íntegra en el Informe 1;
el Informe 2 incluye el resumen necesario para ser leído de forma independiente.

---

## Síntesis por fases

### Fase A — Análisis exploratorio y preprocesamiento

| Requisito | Resultado |
|---|---|
| Dimensiones | 442 × 10, todas `float64` |
| Limpieza / nulos | **0 nulos, 0 duplicados** → no se requiere imputación |
| Outliers | 2 a 9 por variable (IQR); **ninguno en el target**. Se conservan: control winsorizado 1–99 % mueve el R² solo **+0.0016** |
| Multicolinealidad | **Severa y estructural.** VIF: `s1` 59.20, `s2` 39.19, `s3` 15.40, `s5` 10.08. r(`s1`,`s2`) = **+0.897** |
| Estandarización | `StandardScaler` dentro del `Pipeline` (evita fuga de información en la validación cruzada) |
| Partición | 331 entrenamiento / 111 prueba (75 / 25) |

**Causa de la multicolinealidad:** `s4` = `s1`/`s3` por definición y `s2` (LDL) es un
componente de `s1` (colesterol total). No es casual, es una dependencia algebraica.

### Fase B — Modelamiento y comparativa

| Modelo | Términos | R² entren. | R² prueba | R² CV 5-fold |
|---|---|---|---|---|
| **Lineal múltiple — OLS** | 10 | 0.5190 | 0.4849 | **0.4606 ± 0.085** |
| Lineal — sin `s1`,`s2` (VIF) | 8 | 0.5078 | 0.4964 | 0.4504 |
| Lineal — Ridge (α = 1.59) | 10 | 0.5185 | 0.4862 | 0.4597 |
| Lineal — Lasso (α = 0.114) | 10 | 0.5185 | 0.4868 | 0.4577 |
| Polinomial gr. 2 — **sin regularizar** | 65 | 0.6048 | 0.4242 | **0.2979** |
| Polinomial gr. 2 — Ridge (α = 25.12) | 65 | 0.5373 | 0.4911 | **0.4611 ± 0.083** |
| Polinomial gr. 3 — **sin regularizar** | 285 | 0.8960 | **−8.0873** | **−171.12** |
| Polinomial gr. 3 — Ridge (α = 199.53) | 285 | 0.5361 | 0.4942 | 0.4603 |

### Fase C — Aplicativo web

`index.html` + `model.js`, en HTML y JavaScript puro, sin dependencias ni build.
Incorpora **ambos modelos con un selector**, entrada dinámica de las 10 variables,
cálculo paso a paso y las tablas de resultados de las Fases A y B.

Paridad con scikit-learn verificada con precisión de punto flotante en los dos modelos
(p. ej. medianas del dataset: Python `155.862491` → JavaScript `155.8624910825692`).

---

## Comparativa global entre metodologías

| | Lineal Múltiple | Polinomial gr. 2 + Ridge |
|---|---|---|
| Términos | **10** | 65 |
| R² validación cruzada | 0.4606 | 0.4611 |
| Desviación entre particiones | ±0.085 | ±0.083 |
| Brecha entrenamiento − CV | **0.058** | 0.076 |
| RMSE / MAE prueba | 53.37 / 41.55 | 53.05 / 41.59 |
| Coeficientes interpretables | **Sí** | No |
| Depende de hiperparámetro | **No** | Sí (α) |

La diferencia en validación cruzada es de **+0.0005**, frente a una desviación entre
particiones de **±0.083**: la ventaja aparente es **166 veces menor que el ruido de la
propia medición**. Los modelos son **estadísticamente indistinguibles**.

### Modelo recomendado: **Regresión Lineal Múltiple (OLS, 10 variables)**

Por el principio de parsimonia: mismo rendimiento con seis veces menos parámetros,
coeficientes interpretables, menor brecha train−CV y sin dependencia de un
hiperparámetro que calibrar.

---

## Conclusiones técnicas

1. **El dataset es adecuado** para ambas metodologías: variable objetivo continua, 10
   predictores numéricos, sin nulos ni duplicados. Se usó `scaled=False` para poder
   aplicar el preprocesamiento exigido y construir una interfaz con unidades clínicas
   reales.

2. **La multicolinealidad estructural es el hallazgo dominante de la Fase A** y
   condiciona ambas metodologías. En la múltiple infla los errores estándar entre 5 y 7
   veces (`s1`: 22.21 frente a 3.32 de `age`), dejando los coeficientes lipídicos sin
   interpretación individual. En la polinomial explica por qué la expansión no aporta:
   genera términos redundantes a partir de variables ya redundantes.

3. **La regularización es innecesaria en la múltiple e imprescindible en la
   polinomial.** Con 10 variables y 331 muestras no hay margen para sobreajustar, y
   Ridge/Lasso no mueven las métricas. Sobre 65 o 285 términos, en cambio, evitan el
   colapso del modelo.

4. **El sobreajuste quedó cuantificado de forma inequívoca.** El polinomial de grado 3
   sin regularizar alcanza el **mejor R² de entrenamiento del estudio (0.8960)** y es el
   **peor modelo de todos** fuera de la muestra (R² de prueba −8.09, CV −171.12). Es el
   argumento central contra evaluar un modelo por su ajuste en entrenamiento.

5. **La relación es esencialmente lineal en el rango observado.** Dos análisis
   independientes coinciden: el diagnóstico de residuos del modelo lineal no muestra
   patrón estructural, y la expansión polinomial no encuentra nada que capturar.

6. **El techo de R² ≈ 0.46–0.49 proviene de los datos, no del modelo.** Son 442
   pacientes y 10 variables basales, sin información sobre genética, tratamiento,
   adherencia, dieta ni comorbilidades. La literatura original de Efron et al. trabaja
   en este mismo rango. Aumentar la complejidad del modelo no puede compensar una
   carencia de información.

7. **La validación cruzada fue determinante.** Escogiendo por R² de prueba se habría
   seleccionado el modelo sin `s1`,`s2` (0.4964) o el polinomial de grado 3 con Ridge
   (0.4942); la CV demuestra que ninguna de esas diferencias supera el ruido de
   muestreo con 111 muestras de prueba.

---

## Estructura de carpetas y archivos entregados

El caso se organiza en **dos carpetas, una por metodología**, cada una autocontenida
y ejecutable de forma independiente:

```
CASO_3/
├── index.html                        · portada del caso, enlaza ambas metodologías
├── INFORME_CONSOLIDADO.docx / .pdf   · este documento
├── INFORME.md                        · fuente Markdown del consolidado
├── PLAN.md                           · planificación y evaluación del dataset
├── md_a_docx.py                      · conversor de Markdown a Word
│
├── 1_REGRESION_MULTIPLE/
│   ├── train.py                      · Fases A y B, exporta el modelo
│   ├── model.js                      · 10 coeficientes + intercepto
│   ├── index.html                    · aplicativo web interactivo
│   ├── resultados.json               · métricas en formato legible por máquina
│   ├── INFORME_1_REGRESION_MULTIPLE.docx / .pdf / .md
│   └── figuras/                      · 7 gráficos (boxplots, correlación, VIF,
│                                        residuos, real vs. predicho, coeficientes)
│
└── 2_REGRESION_POLINOMIAL/
    ├── train.py                      · Fases A y B, exporta el modelo
    ├── model.js                      · 65 términos + exponentes de la expansión
    ├── index.html                    · aplicativo web interactivo
    ├── resultados.json               · métricas en formato legible por máquina
    ├── INFORME_2_REGRESION_POLINOMIAL.docx / .pdf / .md
    └── figuras/                      · 3 gráficos (correlación, curva de validación,
                                         brecha entrenamiento-validación)
```

Cada carpeta se reproduce por completo con:

```bash
python train.py
```

Los aplicativos se sirven por HTTP (el `<script src="model.js">` no carga bajo
`file://` en algunos navegadores):

```bash
python -m http.server 8000
```

## Referencias

1. Efron, B., Hastie, T., Johnstone, I. y Tibshirani, R. (2004). *Least Angle
   Regression*. **Annals of Statistics**, 32(2), 407–499.
   https://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf
2. Pedregosa, F. *et al.* (2011). *Scikit-learn: Machine Learning in Python*.
   **Journal of Machine Learning Research**, 12, 2825–2830.
3. Documentación de scikit-learn — `sklearn.datasets.load_diabetes`.
   https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
4. James, G., Witten, D., Hastie, T. y Tibshirani, R. (2021). *An Introduction to
   Statistical Learning*, 2.ª ed. Springer.
5. Montgomery, D. C., Peck, E. A. y Vining, G. G. (2021). *Introduction to Linear
   Regression Analysis*, 6.ª ed. Wiley.
6. Fuente original de los datos:
   https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
