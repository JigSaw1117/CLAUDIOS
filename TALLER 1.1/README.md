# Taller 1 — Modelado Predictivo Multisectorial

**Universidad Andina del Cusco** · Facultad de Ingeniería — Escuela Profesional de Ingeniería de Sistemas
Asignatura: Inteligencia Artificial / Aprendizaje Automático · Semestre 2026-II

Regresión Lineal Múltiple, Regresión Polinomial y despliegue de aplicativos de IA sobre
tres dominios distintos: **inmobiliario**, **agroindustrial** y **salud**.

🔗 **Aplicativo desplegado:** https://californication1.netlify.app/
🔗 **Repositorio:** https://github.com/JigSaw1117/CLAUDIOS

---

## Los tres casos de estudio

| | Caso 1 — Housing | Caso 2 — Vino | Caso 3 — Diabetes |
|---|---|---|---|
| **Dataset** | `housing.csv` (California, 1990) | `WineQT.csv` | `load_diabetes` (Scikit-learn) |
| **Registros** | 20 640 distritos | 1 143 vinos | 442 pacientes |
| **Predictores** | 8 numéricos + 1 categórico | 11 físico-químicos | 10 clínicos |
| **Objetivo** | Valor mediano de vivienda (US$) | Calidad sensorial (0–10) | Progresión de la enfermedad |
| **Partición** | 80 / 20 | 80 / 20 | 75 / 25 |
| **Modelo recomendado** | Polinómica grado 2 | **Lineal múltiple** | **Lineal múltiple** |

### Resultados

| Caso | Modelo | Términos | R² entren. | R² prueba | R² CV 5-fold |
|---|---|---|---|---|---|
| 1 | Lineal múltiple | 13 | 0.6497 | 0.6254 | **0.6477** ± 0.013 |
| 1 | Polinómica gr. 2 | 49 | 0.7053 | **0.6570** | 0.6262 ± 0.133 |
| 2 | **Lineal múltiple** | 11 | 0.3822 | **0.3171** | **0.3462** ± 0.043 |
| 2 | Polinómica gr. 2 | 77 | 0.4591 | 0.2809 | 0.2316 ± 0.147 |
| 3 | **Lineal múltiple** | 10 | 0.5190 | 0.4849 | **0.4606** ± 0.085 |
| 3 | Polinomial gr. 2 + Ridge | 65 | 0.5373 | 0.4911 | 0.4611 ± 0.083 |
| 3 | Polinomial gr. 3 sin regularizar | 285 | 0.8960 | **−8.0873** | **−171.12** |

La última fila es el resultado más instructivo del taller: **el modelo con mejor ajuste en
entrenamiento es el peor fuera de la muestra**. Un R² negativo significa que predice peor
que responder siempre la media.

---

## Estructura del repositorio

```
CLAUDIOS/
├── index.html                    · aplicativo de página única con botones por caso
├── portada.html                  · portada institucional (vista de inicio)
├── requirements.txt · netlify.toml · README.md
│
├── CASO_1/                       · California Housing
│   ├── CASO_1_analisis.ipynb     · cuaderno: Fases A y B completas
│   ├── entrenar.py               · entrena la polinómica → modelo.json + .pkl
│   ├── analisis_fases_a_b.py     · outliers, VIF y comparativa lineal vs polinómica
│   ├── interfaz.py               · interfaz de escritorio (Tkinter)
│   ├── index.html · modelo.json  · aplicativo web (polinómica)
│   ├── housing.csv
│   └── 1_REGRESION_MULTIPLE/     · metodología lineal múltiple
│       ├── train.py · model.js · index.html · resultados.json · figuras/
│
├── CASO_2/                       · Wine Quality
│   ├── CASO_2_analisis.ipynb     · cuaderno: Fases A y B completas
│   ├── train_wine_model.py       · entrena ambos modelos → wine_model_*.json
│   ├── analisis_fases_a_b.py     · outliers, VIF y comparativa
│   ├── index.html                · aplicativo web (selector lineal / polinómica)
│   ├── wine_model_linear.json · wine_model_poly.json
│   └── WineQT.csv
│
└── CASO_3/                       · Diabetes
    ├── CASO_3_analisis.ipynb     · cuaderno: Fases A y B completas
    ├── index.html                · portada del caso
    ├── INFORME_CONSOLIDADO.pdf / .docx · md_a_docx.py · PLAN.md
    ├── 1_REGRESION_MULTIPLE/     · train.py · model.js · index.html · figuras/ · informe
    └── 2_REGRESION_POLINOMIAL/   · train.py · model.js · index.html · figuras/ · informe
```

---

## Cómo reproducirlo

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Ejecutar los cuadernos

```bash
jupyter lab
```

Cada `CASO_N/CASO_N_analisis.ipynb` recorre las Fases A y B de su caso: limpieza, nulos,
outliers, matriz de correlación, VIF, estandarización, y el entrenamiento y comparativa de
**Regresión Lineal Múltiple frente a Regresión Polinomial**. Todos usan `random_state=42`,
de modo que los resultados son reproducibles al dígito.

### 3. Reentrenar los modelos

```bash
cd CASO_1                     && python entrenar.py            # polinómica → modelo.json
cd CASO_1/1_REGRESION_MULTIPLE && python train.py              # lineal múltiple → model.js
cd CASO_2                     && python train_wine_model.py    # ambos → wine_model_*.json
cd CASO_3/1_REGRESION_MULTIPLE && python train.py
cd CASO_3/2_REGRESION_POLINOMIAL && python train.py
```

### 4. Levantar el aplicativo en local

```bash
python -m http.server 8000
```

Y abrir http://localhost:8000.

> **Importante:** el sitio **debe servirse por HTTP**. Abrir los `.html` con doble clic
> (`file://`) falla, porque los navegadores bloquean `fetch()` bajo ese protocolo y las
> páginas cargan su modelo desde archivos externos.

---

## Arquitectura del aplicativo

`index.html` es una **aplicación de página única**: una barra superior con botones alterna
entre las siete vistas (Inicio, Caso 1 múltiple, Caso 1 polinómica, Caso 2, Caso 3 resumen,
Caso 3 múltiple, Caso 3 polinomial) sin recargar ni abandonar la página.

**Ningún modelo está incrustado en el HTML.** Cada página consume su modelo entrenado desde
un archivo externo:

| Página | Consume | Mecanismo |
|---|---|---|
| Caso 1 · Polinómica | `modelo.json` | `fetch()` |
| Caso 1 · Múltiple | `model.js` | `<script src>` |
| Caso 2 | `wine_model_linear.json` + `wine_model_poly.json` | `fetch()` |
| Caso 3 · Múltiple | `model.js` | `<script src>` |
| Caso 3 · Polinomial | `model.js` | `<script src>` |

Python entrena y exporta las medias, desviaciones, exponentes de la expansión polinómica,
coeficientes e intercepto; JavaScript reproduce el mismo pipeline de preprocesamiento y
predicción. **La paridad numérica se verificó frente a scikit-learn con precisión de punto
flotante**, por ejemplo en el Caso 3: Python `155.862491` → JavaScript `155.8624910825692`.

La ventaja es que no hace falta servidor de inferencia: el sitio es estático, se despliega en
cualquier hosting y la predicción es instantánea en el navegador.

---

## Metodología

### Fase A — Análisis exploratorio y preprocesamiento

Aplicada a los tres datasets: limpieza, tratamiento de nulos, detección de outliers por el
método IQR, análisis de multicolinealidad (matriz de correlación de Pearson y **VIF**) y
estandarización con `StandardScaler`.

El escalado y la imputación van **dentro de un `Pipeline` de scikit-learn**. No es un detalle
cosmético: así los parámetros se recalculan con los datos de entrenamiento de cada partición
de la validación cruzada, evitando la fuga de información hacia el conjunto de prueba.

**Hallazgos de multicolinealidad por caso:**

- **Caso 1** — VIF severo en `households` (28.3), `total_bedrooms` (26.9) y `total_rooms`
  (12.1): las cuatro variables de tamaño del distrito miden lo mismo.
- **Caso 2** — Ningún VIF supera 10. Es el único dataset sin multicolinealidad severa.
- **Caso 3** — VIF de `s1` = 59.2 y `s2` = 39.2. La causa es **algebraica**: `s4` = `s1`/`s3`
  por definición y `s2` (LDL) es un componente de `s1` (colesterol total). El efecto se mide
  en la Fase B: los errores estándar de los coeficientes lipídicos son entre 5 y 7 veces
  mayores, por lo que no son interpretables individualmente.

### Fase B — Modelamiento y comparativa

Los dos modelos de cada caso comparten **exactamente** el mismo preprocesamiento y la misma
partición; la única diferencia entre ellos es la expansión polinómica. Así, cualquier
diferencia de rendimiento es atribuible al modelo y no al tratamiento de los datos.

Se reportan tres medidas: R² de entrenamiento, R² de prueba y **R² de validación cruzada
5-fold**, esta última la más fiable por promediar cinco particiones en lugar de depender de
una sola.

### Fase C — Aplicativo web

> **Nota sobre la tecnología.** El enunciado sugiere Streamlit, Gradio o Flask/FastAPI. Se optó
> por **HTML + JavaScript puro** por una razón concreta: al reimplementar el pipeline de
> inferencia en el cliente, el aplicativo no necesita servidor de Python, arranca al instante,
> no depende de un servicio que pueda hibernar y se despliega en cualquier hosting estático.
> La lógica de inferencia está aislada en funciones (`expandir()`, `predecir()`) que un port a
> Streamlit reutilizaría directamente cargando los mismos `.json`.

---

## Conclusiones

1. **La utilidad de la regresión polinomial depende del tamaño muestral.** Con 20 640
   registros (Caso 1) aporta una mejora medible; con 1 143 (Caso 2) ya sobreajusta; con 442
   y multicolinealidad estructural (Caso 3) no encuentra nada que capturar.

2. **El R² alcanzable lo limitan los datos, no la complejidad del modelo.** El techo de ≈0.32
   en vino proviene de una variable objetivo subjetiva y discreta; el de ≈0.46 en diabetes, de
   la ausencia de variables clínicas relevantes (genética, tratamiento, adherencia).

3. **La regularización es innecesaria cuando sobran datos e imprescindible cuando faltan.** En
   el Caso 3, Ridge es lo único que evita el colapso del polinomio de grado 3.

4. **La multicolinealidad debe diagnosticarse antes de interpretar coeficientes**, y no impide
   predecir: son dos cosas distintas.

5. **La selección de modelos se guió por validación cruzada y parsimonia**, no por el mejor
   número aislado. En dos de los tres dominios gana el modelo más simple.

---

## Referencias

1. Efron, B., Hastie, T., Johnstone, I. y Tibshirani, R. (2004). *Least Angle Regression*.
   **Annals of Statistics**, 32(2), 407–499.
2. Cortez, P., Cerdeira, A., Almeida, F., Matos, T. y Reis, J. (2009). *Modeling wine
   preferences by data mining from physicochemical properties*. **Decision Support Systems**,
   47(4), 547–553.
3. Pace, R. K. y Barry, R. (1997). *Sparse spatial autoregressions*. **Statistics & Probability
   Letters**, 33(3), 291–297.
4. Pedregosa, F. *et al.* (2011). *Scikit-learn: Machine Learning in Python*. **JMLR**, 12,
   2825–2830.
5. James, G., Witten, D., Hastie, T. y Tibshirani, R. (2021). *An Introduction to Statistical
   Learning* (2.ª ed.). Springer.
6. Montgomery, D. C., Peck, E. A. y Vining, G. G. (2021). *Introduction to Linear Regression
   Analysis* (6.ª ed.). Wiley.
