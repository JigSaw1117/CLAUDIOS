# Caso 3 — Diabetes: plan de trabajo

**Contexto del problema (según el enunciado):** predecir la progresión cuantitativa
de la enfermedad un año después del inicio, a partir de 10 variables basales
(edad, IMC, presión arterial, etc.).

**Tipo de problema:** regresión (el objetivo es una magnitud continua), no clasificación.
Es lo que corresponde a Regresión Lineal Múltiple y Regresión Polinomial.

---

## 0. ¿Qué dataset usar? ¿Es adecuado?

**Sí, es adecuado — pero hay que usar el de Scikit-learn, no el de Kaggle.**

El cuadro del enunciado dice literalmente **"Diabetes (Scikit-learn)"**. Ese es el
dataset de Efron et al. (2004), incluido en la librería:

```python
from sklearn.datasets import load_diabetes
```

Coincide **exactamente** con el contexto del enunciado: 442 pacientes, 10 variables
basales, y el target es *"una medida cuantitativa de la progresión de la enfermedad
un año después del inicio"*.

### Advertencia sobre el dataset de Kaggle de la imagen

`imtkaggleteam/diabetes` (11 kB) **no lo pude verificar** — descargarlo requiere claves
de API de Kaggle. Existe un riesgo real de que sea el *Pima Indians Diabetes*, que es
un problema **de clasificación binaria** (columna `Outcome` = 0/1, tiene/no tiene
diabetes). Si fuera ese, **no serviría**: no se puede hacer regresión lineal sobre una
etiqueta binaria, y no habría ninguna "progresión cuantitativa" que predecir.

Como el enunciado dice "(Scikit-learn)", la ruta segura es `load_diabetes()`: no
necesita descarga, ni claves de API, y es 100% reproducible para quien clone el repo.
Si quieres, descarga el ZIP de Kaggle manualmente y lo comparo antes de decidir.

### Detalle crítico: `scaled=True` vs `scaled=False`

`load_diabetes()` devuelve por defecto los datos **ya centrados y escalados**
(valores adimensionales entre −0.14 y 0.20). Eso rompe dos requisitos:

- **Fase A** pide aplicar StandardScaler/MinMaxScaler — no tiene sentido escalar algo
  ya escalado, y el informe no podría mostrar el paso.
- **Fase C** pide que el usuario ingrese parámetros — nadie puede escribir
  "IMC = 0.0507"; tiene que escribir "IMC = 26.4".

**Por eso usaremos `load_diabetes(scaled=False)`**, que da las unidades clínicas
reales, y aplicaremos nuestro propio `StandardScaler` ajustado solo con el set de
entrenamiento.

---

## 1. Tipos de datos (verificado ejecutando el dataset)

442 filas × 10 columnas. **Todas `float64`. Cero nulos. Cero duplicados.**

| Variable | Significado | Unidad | Media | Rango |
|---|---|---|---|---|
| `age` | Edad | años | 48.5 | 19 – 79 |
| `sex` | Sexo | **categórica binaria** (1 / 2) | 1.47 | 1 – 2 |
| `bmi` | Índice de masa corporal | kg/m² | 26.4 | 18.0 – 42.2 |
| `bp`  | Presión arterial media | mm Hg | 94.6 | 62 – 133 |
| `s1`  | Colesterol total (TC) | mg/dL | 189.1 | 97 – 301 |
| `s2`  | Lipoproteínas de baja densidad (LDL) | mg/dL | 115.4 | 41.6 – 242.4 |
| `s3`  | Lipoproteínas de alta densidad (HDL) | mg/dL | 49.8 | 22 – 99 |
| `s4`  | Razón colesterol total / HDL | ratio | 4.07 | 2.0 – 9.09 |
| `s5`  | Triglicéridos séricos, *posiblemente* en log | log | 4.64 | 3.26 – 6.11 |
| `s6`  | Glucosa en sangre | mg/dL | 91.3 | 58 – 124 |

**Target:** progresión de la enfermedad a un año. Continuo, 25 – 346, media 152.1,
asimetría 0.44 (aceptablemente simétrico → **no hace falta transformación log**,
a diferencia del caso California).

Los nombres `s1`–`s6` salen del `.DESCR` oficial (`tc`, `ldl`, `hdl`, `tch`, `ltg`,
`glu`). La propia documentación de scikit-learn advierte que el significado exacto
*"puede no ser claro, especialmente para `ltg`"*, porque el dataset original no lo
documenta de forma explícita. Conviene decirlo así en el informe.

Ojo con `sex`: es la única **categórica**. Está codificada 1/2, no 0/1. Hay que
recodificarla a 0/1 para que el coeficiente sea interpretable ("efecto de ser sexo 2
respecto a sexo 1"), y **no debe interpretarse como continua** en el informe.

---

## Fase A — Análisis exploratorio y preprocesamiento

### A.1 Limpieza y nulos

No hay nulos ni duplicados. **Esto se documenta explícitamente** (`.isna().sum()`,
`.duplicated().sum()`) para dejar constancia de que se verificó, no de que se omitió.

### A.2 Outliers

Medidos por IQR (1.5×): `s6` 9, `s1` 8, `s2` 7, `s3` 7, `s5` 4, `bmi` 3, `s4` 2.
El **target no tiene outliers**.

Son pocos (máx. 2% por variable) y son **valores clínicos plausibles** (un colesterol
de 301 mg/dL es alto, pero real). **Decisión: no eliminarlos.** Con 442 muestras,
borrar filas cuesta más de lo que aporta. Se documentan con boxplots y se justifica
la decisión. Se compara contra un winsorizado al percentil 1–99 para demostrar en el
informe que el efecto es despreciable.

### A.3 Multicolinealidad — *el hallazgo central de este caso*

Matriz de correlación y VIF (ya calculados):

| Variable | VIF | | Par | r |
|---|---|---|---|---|
| `s1` | **59.20** | | `s1`–`s2` | **+0.897** |
| `s2` | **39.19** | | `s3`–`s4` | −0.738 |
| `s3` | 15.40 | | `s2`–`s4` | +0.660 |
| `s5` | 10.08 | | `s4`–`s5` | +0.618 |
| `s4` | 8.89  | | `s1`–`s4` | +0.542 |
| resto | < 1.6 | | `s1`–`s5` | +0.516 |

VIF > 10 indica multicolinealidad severa. **Y tiene explicación física:** `s4` es por
definición la razón `s1`/`s3`, y `s2` (LDL) es un componente de `s1` (colesterol
total). O sea, las variables lipídicas se contienen unas a otras. Este es un caso de
manual para el informe.

**Tratamiento:** se probarán y compararán tres rutas —
(a) eliminar `s1` y `s2` por VIF, (b) Ridge (L2), (c) Lasso (L1, selección automática).

### A.4 Estandarización

`StandardScaler` **ajustado solo con el 75% de entrenamiento** y aplicado a ambos sets,
dentro de un `Pipeline` de scikit-learn para que no haya fuga de información.

---

## Fase B — Modelamiento y comparativa

**Partición 75 / 25** (331 train / 111 test), `random_state=42`, consistente con el
Caso California.

> Con solo 111 muestras de prueba, el R² de test es ruidoso. Por eso **toda métrica se
> reporta junto a validación cruzada 5-fold** sobre el set de entrenamiento. Es lo que
> hace defendible la comparación.

### B.1 Regresión Lineal Múltiple — resultados ya medidos

| Modelo | R² train | R² test | R² CV 5-fold |
|---|---|---|---|
| OLS, 10 variables | 0.5190 | 0.4849 | 0.4606 |
| Ridge (α por CV) | 0.5185 | 0.4862 | 0.4585 |
| Lasso (α por CV) | 0.5185 | 0.4868 | 0.4577 |
| **OLS sin `s1`,`s2` (corte por VIF)** | 0.5078 | **0.4964** | 0.4504 |

Brecha train–test ≈ 0.03 → **no hay overfitting**. Quitar las dos variables de VIF
más alto **mejora** el test y deja el modelo interpretable: ese será el modelo base.

### B.2 Regresión Polinomial (fase siguiente) — advertencia ya verificada

`PolynomialFeatures(degree=2)` sobre las 10 variables genera 65 términos y
**sobreajusta**:

| Modelo | R² train | R² test | R² CV |
|---|---|---|---|
| Lineal múltiple | 0.5190 | 0.4849 | **0.4606** |
| Polinomial gr. 2 (sin regularizar) | 0.6048 | 0.4242 | **0.2979** |

El R² de entrenamiento sube y el de validación se **desploma**: overfitting de libro.
Lejos de ser un problema, **este es el mejor material del trabajo** — es exactamente
la comparativa que pide la Fase B, con evidencia numérica. La ruta será: polinomio de
grado 2 **con Ridge/Lasso** y grado elegido por CV, mostrando la curva de validación.

### Sobre la precisión esperada — importante

**El techo realista de este dataset es R² ≈ 0.50.** No es un defecto del modelo: son
442 pacientes, la progresión de una enfermedad depende de factores no medidos
(genética, tratamiento, adherencia, dieta), y la literatura original de Efron et al.
que introdujo este dataset se mueve en ese mismo rango. Un R² de 0.85 aquí sería señal
de fuga de datos, no de buen trabajo. Lo digo por adelantado para que no sorprenda.

---

## Fase C — Aplicativo web

El requisito pide **Streamlit, Gradio o Flask/FastAPI** con menú para los 3 casos.

**Implicación de alcance:** la app actual del Caso California es HTML+JavaScript, y
**no cumple** ese requisito. Habrá que portarla a Streamlit para integrarla en la misma
app multi-caso. Se puede reusar el modelo ya entrenado, así que el coste es bajo.

Propuesta: **Streamlit** (el más simple de desplegar en Streamlit Community Cloud,
que es gratis y está en la lista permitida), con `st.sidebar.radio` para elegir caso y
un `st.form` con sliders acotados al rango real de cada variable.

---

## Estructura de carpetas propuesta

```
repo/
├── README.md                  · descripción global + enlaces
├── requirements.txt
├── app.py                     · Streamlit multi-caso (3 casos)
├── california/                · Caso 1 (ya avanzado)
└── Diabetes/                  · Caso 3
    ├── PLAN.md                · este documento
    ├── notebooks/
    │   ├── 01_eda.ipynb       · Fase A: nulos, outliers, correlación, VIF
    │   └── 02_modelos.ipynb   · Fase B: múltiple + polinomial
    ├── train.py               · script reproducible que exporta el modelo
    └── modelo_diabetes.pkl    · modelo entrenado para la app
```

El enunciado exige `.ipynb` **y** `.py`: los notebooks llevan el análisis narrado con
gráficos, y `train.py` reproduce el entrenamiento final de forma automatizable.

---

## Orden de ejecución

1. **Fase A** — notebook `01_eda.ipynb`: tipos, nulos, outliers, correlación, VIF,
   estandarización. Con gráficos para el informe.
2. **Fase B.1 — Regresión Lineal Múltiple** ← *empezamos aquí, según lo acordado*
3. Fase B.2 — Regresión Polinomial y comparativa.
4. Fase C — app Streamlit e integración de los 3 casos.

## Pendientes por definir

- **Caso 2**: no se ha indicado cuál es. La app de la Fase C debe alojar los 3.
- **Kaggle vs Scikit-learn**: confirmar con el docente, o descargar el ZIP para comparar.
- Integrantes y división de funciones para la carátula del informe.
