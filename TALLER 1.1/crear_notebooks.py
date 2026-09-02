"""
Genera los cuadernos Jupyter (.ipynb) exigidos por el entregable, uno por caso,
y los ejecuta para que queden con sus salidas guardadas.

Cada cuaderno recorre la Fase A (limpieza, nulos, outliers, multicolinealidad y
estandarizacion) y la Fase B (Regresion Lineal Multiple y Regresion Polinomica),
con la narrativa en celdas de texto.

Ejecutar:  python crear_notebooks.py
"""

import os
import nbformat as nbf
from nbclient import NotebookClient

RAIZ = os.path.dirname(os.path.abspath(__file__))


def cuaderno(celdas, destino, carpeta):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md"
                else nbf.v4.new_code_cell(c[1]) for c in celdas]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    ruta_dir = os.path.join(RAIZ, carpeta)
    print(f"  ejecutando {destino} …", end="", flush=True)
    cliente = NotebookClient(nb, timeout=900, kernel_name="python3",
                            resources={"metadata": {"path": ruta_dir}})
    cliente.execute()
    ruta = os.path.join(ruta_dir, destino)
    nbf.write(nb, ruta)
    print(f" ok ({os.path.getsize(ruta)/1024:.0f} kB)")


CABECERA = """# {titulo}

**Universidad Andina del Cusco — Inteligencia Artificial (2026-II)**
Taller 1: Modelado Predictivo Multisectorial

{contexto}

Este cuaderno recorre las dos fases exigidas por el enunciado:

- **Fase A** — limpieza, tratamiento de nulos, deteccion de outliers, analisis de
  multicolinealidad (matriz de correlacion y VIF) y estandarizacion.
- **Fase B** — entrenamiento y comparativa de **Regresion Lineal Multiple** y
  **Regresion Polinomial**, ambas bajo el mismo protocolo para que la comparacion
  sea legitima.

Semilla fija `random_state=42` en todo el cuaderno: los resultados son reproducibles.
"""

IMPORTS = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

RANDOM_STATE = 42
CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
sns.set_theme(style="whitegrid")
pd.set_option("display.width", 120)
print("Entorno listo")"""

MD_VIF = """### Multicolinealidad: matriz de correlacion y VIF

El **Factor de Inflacion de la Varianza** mide cuanto se infla la varianza del
coeficiente de una variable por estar correlacionada con las demas:

$$\\text{VIF}_j = \\frac{1}{1 - R^2_j}$$

donde $R^2_j$ es el ajuste al regresar la variable $j$ contra todas las otras.
La lectura habitual es: **VIF > 5** moderada, **VIF > 10** severa.

La multicolinealidad **no impide predecir**, pero hace inestables e
ininterpretables los coeficientes individuales."""

FUNC_VIF = '''def calcular_vif(X):
    """VIF_j = 1 / (1 - R2_j), regresando cada variable contra las demas."""
    Z = StandardScaler().fit_transform(X)
    vif = {}
    for j, nombre in enumerate(X.columns):
        otras = [k for k in range(Z.shape[1]) if k != j]
        r2 = LinearRegression().fit(Z[:, otras], Z[:, j]).score(Z[:, otras], Z[:, j])
        vif[nombre] = 1 / (1 - r2)
    return pd.Series(vif).sort_values(ascending=False)


def diagnostico(v):
    return "SEVERA" if v > 10 else ("moderada" if v > 5 else "aceptable")'''

MD_FASE_B = """## Fase B — Modelamiento estadistico y comparativa

Se entrenan las **dos metodologias exigidas** compartiendo exactamente el mismo
preprocesamiento y la misma particion. La **unica** diferencia entre ellas es la
expansion polinomica, de modo que cualquier diferencia de rendimiento es
atribuible al modelo y no al tratamiento de los datos.

- **Regresion Lineal Multiple**: $\\hat{y} = \\beta_0 + \\sum_i \\beta_i z_i$
- **Regresion Polinomica grado 2**: anade terminos cuadraticos $z_i^2$ e
  interacciones $z_i z_j$. Sigue siendo **lineal en los parametros**, que es lo
  que define el metodo.

Se reportan tres medidas: R² de entrenamiento, R² de prueba y **R² de validacion
cruzada 5-fold**. La tercera es la mas fiable, porque promedia cinco particiones
distintas en lugar de depender de una sola."""

FUNC_EVAL = '''def evaluar(nombre, grado, X_tr, X_te, y_tr, y_te, escala_dinero=False):
    """Entrena y evalua un modelo; grado=1 es la lineal multiple."""
    pasos = []
    if grado > 1:
        pasos.append(("poly", PolynomialFeatures(degree=grado, include_bias=False)))
    pasos += [("sc", StandardScaler()), ("modelo", LinearRegression())]
    pipe = Pipeline(pasos).fit(X_tr, y_tr)

    p_tr, p_te = pipe.predict(X_tr), pipe.predict(X_te)
    cv = cross_val_score(pipe, X_tr, y_tr, cv=CV, scoring="r2")
    return {
        "modelo": nombre,
        "terminos": pipe.named_steps["modelo"].coef_.size,
        "R2_train": round(r2_score(y_tr, p_tr), 4),
        "R2_test": round(r2_score(y_te, p_te), 4),
        "R2_CV": round(cv.mean(), 4),
        "CV_std": round(cv.std(), 4),
        "brecha": round(r2_score(y_tr, p_tr) - cv.mean(), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_te, p_te)), 4),
        "MAE": round(mean_absolute_error(y_te, p_te), 4),
    }, pipe'''


# =============================================================================
# CASO 1
# =============================================================================
C1 = [
 ("md", CABECERA.format(
     titulo="Caso 1 — California Housing: tasacion de viviendas",
     contexto="**Contexto.** Predecir el valor mediano de la vivienda por distrito "
              "a partir de los datos del censo de California de 1990 (20 640 distritos): "
              "ubicacion, antiguedad, tamano del distrito, ingreso y proximidad al oceano.")),
 ("code", IMPORTS),
 ("md", "## Fase A — Analisis exploratorio y preprocesamiento\n\n### Carga y tipos de datos"),
 ("code", '''df = pd.read_csv("housing.csv")
print(f"Registros: {len(df)}   Columnas: {len(df.columns)}")
NUMERICAS = ["longitude","latitude","housing_median_age","total_rooms",
             "total_bedrooms","population","households","median_income"]
CATEGORICA, OBJETIVO = "ocean_proximity", "median_house_value"
pd.DataFrame({"dtype": df.dtypes.astype(str), "nulos": df.isna().sum(),
              "unicos": df.nunique()})'''),
 ("md", "### Valores nulos\n\n`total_bedrooms` es la unica columna con faltantes. Se imputan "
        "con la **mediana**, mas robusta que la media ante la fuerte asimetria de la variable. "
        "La imputacion se hara **dentro del Pipeline**, para que la mediana se calcule solo con "
        "los datos de entrenamiento y no se filtre informacion del conjunto de prueba."),
 ("code", '''nulos = df.isna().sum()
print(nulos[nulos > 0].to_string())
print(f"\\nProporcion de nulos en total_bedrooms: {100*df['total_bedrooms'].isna().mean():.2f} %")
print(f"Filas duplicadas: {df.duplicated().sum()}")
print(f"\\nCategorias de {CATEGORICA}: {sorted(df[CATEGORICA].unique())}")'''),
 ("md", "### Deteccion de outliers (metodo IQR)\n\nSe consideran atipicos los valores fuera de "
        "$[Q_1 - 1.5\\,\\text{IQR},\\; Q_3 + 1.5\\,\\text{IQR}]$."),
 ("code", '''resumen = []
for c in NUMERICAS + [OBJETIVO]:
    q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    n = int(((df[c] < lo) | (df[c] > hi)).sum())
    resumen.append({"variable": c, "outliers": n, "%": round(100*n/len(df), 2),
                    "limite_inf": round(lo, 2), "limite_sup": round(hi, 2)})
pd.DataFrame(resumen).set_index("variable")'''),
 ("md", "**Decision: se conservan.** Los atipicos de `total_rooms`, `population` o `households` "
        "son distritos genuinamente grandes, no errores de medicion; eliminarlos sesgaria el "
        "modelo hacia el distrito promedio.\n\n"
        "Aparte, el objetivo esta **censurado**: todos los distritos por encima de US$ 500 000 "
        "se registraron con el mismo valor tope. Es un artefacto administrativo del censo y una "
        "limitacion conocida del dataset."),
 ("code", '''censurados = int((df[OBJETIVO] >= 500001).sum())
print(f"Registros en el tope de 500 001: {censurados} ({100*censurados/len(df):.2f} %)")

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, c in zip(axes.ravel(), NUMERICAS):
    sns.boxplot(y=df[c], ax=ax, color="#0d3c6c", width=.5)
    ax.set_title(c, fontsize=10); ax.set_ylabel("")
fig.suptitle("Fase A — Diagramas de caja", fontsize=13)
plt.tight_layout(); plt.show()'''),
 ("md", MD_VIF),
 ("code", FUNC_VIF),
 ("code", '''corr = df[NUMERICAS].corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True,
            linewidths=.5, ax=ax, cbar_kws={"shrink": .8})
ax.set_title("Matriz de correlacion de Pearson"); plt.tight_layout(); plt.show()

print("Pares con |r| > 0.5:")
for i in range(len(corr)):
    for j in range(i+1, len(corr)):
        if abs(corr.iloc[i, j]) > .5:
            print(f"  {corr.index[i]:<20} - {corr.columns[j]:<20} r = {corr.iloc[i,j]:+.4f}")'''),
 ("code", '''vif = calcular_vif(df[NUMERICAS].fillna(df[NUMERICAS].median()))
tabla_vif = pd.DataFrame({"VIF": vif.round(2), "diagnostico": vif.map(diagnostico)})
display(tabla_vif)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(vif.index, vif.values,
       color=["#d62828" if v > 10 else "#f77f00" if v > 5 else "#0d3c6c" for v in vif])
ax.axhline(10, ls="--", c="#d62828"); ax.axhline(5, ls=":", c="#f77f00")
ax.set_ylabel("VIF"); plt.xticks(rotation=30, ha="right")
ax.set_title("Factor de Inflacion de la Varianza"); plt.tight_layout(); plt.show()'''),
 ("md", "`total_rooms`, `total_bedrooms`, `population` y `households` miden todas el **tamano "
        "del distrito**: de ahi su VIF elevado. Es multicolinealidad esperable y estructural. "
        "No impide predecir, pero desaconseja leer esos coeficientes por separado."),
 ("md", MD_FASE_B),
 ("code", '''from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

X, y = df[NUMERICAS + [CATEGORICA]], df[OBJETIVO]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=RANDOM_STATE)
print(f"Entrenamiento: {len(X_tr)}   Prueba: {len(X_te)}   (80 / 20)")

def pipeline_caso1(grado):
    pasos = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    if grado > 1:
        pasos.append(("poly", PolynomialFeatures(degree=grado, include_bias=False)))
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline(pasos), NUMERICAS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), [CATEGORICA])])),
        ("modelo", LinearRegression())])

def evaluar_caso1(nombre, grado):
    pipe = pipeline_caso1(grado).fit(X_tr, y_tr)
    p_tr, p_te = pipe.predict(X_tr), pipe.predict(X_te)
    cv = cross_val_score(pipe, X_tr, y_tr, cv=CV, scoring="r2")
    return {"modelo": nombre, "terminos": pipe.named_steps["modelo"].coef_.size,
            "R2_train": round(r2_score(y_tr, p_tr), 4),
            "R2_test": round(r2_score(y_te, p_te), 4),
            "R2_CV": round(cv.mean(), 4), "CV_std": round(cv.std(), 4),
            "brecha": round(r2_score(y_tr, p_tr) - cv.mean(), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_te, p_te)), 2),
            "MAE": round(mean_absolute_error(y_te, p_te), 2)}, pipe'''),
 ("md", "### Fase B.1 — Regresion Lineal Multiple"),
 ("code", '''m_lin, pipe_lin = evaluar_caso1("Lineal multiple (OLS)", 1)
pd.Series(m_lin).to_frame("valor")'''),
 ("md", "### Fase B.2 — Regresion Polinomica grado 2"),
 ("code", '''m_poly, pipe_poly = evaluar_caso1("Polinomica grado 2", 2)
pd.Series(m_poly).to_frame("valor")'''),
 ("md", "### Fase B.3 — Comparativa"),
 ("code", '''comp = pd.DataFrame([m_lin, m_poly]).set_index("modelo")
display(comp)
print(f"Ganancia en R2 de prueba : {m_poly['R2_test'] - m_lin['R2_test']:+.4f}")
print(f"Diferencia en MAE        : {m_poly['MAE'] - m_lin['MAE']:+,.2f} US$")
print(f"R2 de validacion cruzada : lineal {m_lin['R2_CV']:.4f} +- {m_lin['CV_std']:.3f}"
      f"  |  polinomica {m_poly['R2_CV']:.4f} +- {m_poly['CV_std']:.3f}")'''),
 ("code", '''fig, ax = plt.subplots(figsize=(7, 4.5))
et = [f"Lineal multiple\\n({m_lin['terminos']} term.)", f"Polinomica gr. 2\\n({m_poly['terminos']} term.)"]
x, an = np.arange(2), .35
ax.bar(x - an/2, [m_lin["R2_train"], m_poly["R2_train"]], an, label="Entrenamiento", color="#0d3c6c")
ax.bar(x + an/2, [m_lin["R2_test"], m_poly["R2_test"]], an, label="Prueba", color="#f77f00")
ax.set_xticks(x); ax.set_xticklabels(et); ax.set_ylabel("R2"); ax.legend()
ax.set_title("Fase B — Comparativa de ajuste"); plt.tight_layout(); plt.show()'''),
 ("md", """### Conclusion del Caso 1

La expansion polinomica **mejora el R² de prueba** (+0.0316) y **reduce el MAE** en unos
US$ 3 689: con 20 640 registros hay muestra suficiente para estimar 49 coeficientes, y las
interacciones entre coordenadas geograficas e ingreso capturan estructura espacial real.

Sin embargo, la lectura no es unanime. En **validacion cruzada** el orden se invierte y,
sobre todo, la **desviacion entre particiones** del polinomio es un orden de magnitud mayor:
es un modelo mas potente pero **mucho menos estable**. La brecha entrenamiento−CV tambien
crece de forma notable.

**Conclusion honesta:** el polinomio es preferible en este dominio por su mejor error de
prueba, pero conviene reportar su inestabilidad en lugar de presentarlo como una victoria
limpia."""),
]

# =============================================================================
# CASO 2
# =============================================================================
C2 = [
 ("md", CABECERA.format(
     titulo="Caso 2 — Wine Quality: calidad del vino tinto",
     contexto="**Contexto.** Predecir la calidad sensorial del vino tinto (escala 0–10, "
              "asignada por catadores) a partir de 11 propiedades fisico-quimicas medidas "
              "en laboratorio, sobre 1 143 muestras.")),
 ("code", IMPORTS),
 ("md", "## Fase A — Analisis exploratorio y preprocesamiento\n\n### Carga y limpieza\n\n"
        "La columna `Id` es un identificador de fila: no aporta informacion predictiva y se elimina."),
 ("code", '''df = pd.read_csv("WineQT.csv")
print(f"Registros: {len(df)}   Columnas originales: {len(df.columns)}")
if "Id" in df.columns:
    df = df.drop(columns=["Id"])
    print("Columna 'Id' eliminada.")
OBJETIVO = "quality"
PREDICTORES = [c for c in df.columns if c != OBJETIVO]
print(f"Predictores: {len(PREDICTORES)}")
df.head()'''),
 ("code", '''print(f"Nulos totales: {df.isna().sum().sum()}")
print(f"Duplicados: {df.duplicated().sum()}")
display(df.describe().T[["mean","std","min","50%","max"]].round(3))
print("\\nDistribucion de la calidad (variable objetivo):")
print(df[OBJETIVO].value_counts().sort_index().to_string())'''),
 ("md", "La calidad es **discreta y muy desbalanceada**: se concentra en 5 y 6. Esto acota de "
        "antemano el R² alcanzable por cualquier modelo de regresion."),
 ("md", "### Deteccion de outliers (metodo IQR)"),
 ("code", '''resumen = []
for c in PREDICTORES:
    q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
    n = int(((df[c] < q1-1.5*iqr) | (df[c] > q3+1.5*iqr)).sum())
    resumen.append({"variable": c, "outliers": n, "%": round(100*n/len(df), 2)})
tabla = pd.DataFrame(resumen).set_index("variable").sort_values("outliers", ascending=False)
display(tabla)
print(f"Total de celdas atipicas: {tabla['outliers'].sum()} de {len(df)*len(PREDICTORES)}")'''),
 ("code", '''fig, axes = plt.subplots(3, 4, figsize=(16, 9))
for ax, c in zip(axes.ravel(), PREDICTORES):
    sns.boxplot(y=df[c], ax=ax, color="#8b1a35", width=.5)
    ax.set_title(c, fontsize=9); ax.set_ylabel("")
for ax in axes.ravel()[len(PREDICTORES):]:
    ax.axis("off")
fig.suptitle("Fase A — Diagramas de caja", fontsize=13)
plt.tight_layout(); plt.show()'''),
 ("md", "**Decision: se conservan.** Son composiciones quimicas reales; un vino puede tener "
        "acidez volatil alta y seguir siendo un vino valido. Con solo 1 143 muestras, eliminarlos "
        "reduciria una muestra ya pequena y sesgaria el modelo hacia el vino promedio."),
 ("md", MD_VIF),
 ("code", FUNC_VIF),
 ("code", '''corr = df[PREDICTORES].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True,
            linewidths=.5, ax=ax, cbar_kws={"shrink": .8}, annot_kws={"size": 7})
ax.set_title("Matriz de correlacion de Pearson"); plt.tight_layout(); plt.show()

print("Pares con |r| > 0.5:")
for i in range(len(corr)):
    for j in range(i+1, len(corr)):
        if abs(corr.iloc[i, j]) > .5:
            print(f"  {corr.index[i]:<22} - {corr.columns[j]:<22} r = {corr.iloc[i,j]:+.4f}")'''),
 ("code", '''vif = calcular_vif(df[PREDICTORES])
display(pd.DataFrame({"VIF": vif.round(2), "diagnostico": vif.map(diagnostico)}))

print("\\nCorrelacion de cada propiedad con la calidad:")
print(df[PREDICTORES + [OBJETIVO]].corr()[OBJETIVO].drop(OBJETIVO)
        .sort_values(ascending=False).round(4).to_string())'''),
 ("md", "Ningun VIF supera 10: **no hay multicolinealidad severa** en este dataset, a diferencia "
        "de los casos 1 y 3. `alcohol` (+0.485) y `volatile acidity` (−0.407) son las propiedades "
        "mas asociadas a la calidad."),
 ("md", MD_FASE_B),
 ("code", FUNC_EVAL),
 ("code", '''X, y = df[PREDICTORES], df[OBJETIVO]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=RANDOM_STATE)
print(f"Entrenamiento: {len(X_tr)}   Prueba: {len(X_te)}   (80 / 20)")'''),
 ("md", "### Fase B.1 — Regresion Lineal Multiple"),
 ("code", '''m_lin, pipe_lin = evaluar("Lineal multiple (OLS)", 1, X_tr, X_te, y_tr, y_te)
display(pd.Series(m_lin).to_frame("valor"))

co = pd.Series(pipe_lin.named_steps["modelo"].coef_, index=PREDICTORES)
print("Coeficientes (variables estandarizadas):")
print(co.reindex(co.abs().sort_values(ascending=False).index).round(4).to_string())'''),
 ("md", "### Fase B.2 — Regresion Polinomica grado 2\n\n"
        "Con 11 variables, la expansion de grado 2 genera 77 terminos: 11 lineales, "
        "11 cuadraticos y 55 interacciones."),
 ("code", '''m_poly, pipe_poly = evaluar("Polinomica grado 2", 2, X_tr, X_te, y_tr, y_te)
pd.Series(m_poly).to_frame("valor")'''),
 ("md", "### Fase B.3 — Comparativa"),
 ("code", '''comp = pd.DataFrame([m_lin, m_poly]).set_index("modelo")
display(comp)

fig, ax = plt.subplots(figsize=(7, 4.5))
et = [f"Lineal multiple\\n({m_lin['terminos']} term.)", f"Polinomica gr. 2\\n({m_poly['terminos']} term.)"]
x, an = np.arange(2), .35
ax.bar(x - an/2, [m_lin["R2_train"], m_poly["R2_train"]], an, label="Entrenamiento", color="#8b1a35")
ax.bar(x + an/2, [m_lin["R2_test"], m_poly["R2_test"]], an, label="Prueba", color="#f77f00")
ax.set_xticks(x); ax.set_xticklabels(et); ax.set_ylabel("R2"); ax.legend()
ax.set_title("Fase B — El sobreajuste, visible"); plt.tight_layout(); plt.show()'''),
 ("md", """### Conclusion del Caso 2

Caso de sobreajuste **de manual**: al pasar de 11 a 77 terminos el R² de entrenamiento
**sube** (+0.077) mientras el de prueba **baja** (−0.036) y el de validacion cruzada se
hunde. La brecha entrenamiento−CV se multiplica por seis.

Con 914 muestras de entrenamiento, 77 parametros son demasiados: el modelo memoriza el
ruido de la muestra en lugar de aprender la relacion.

**Modelo recomendado: Regresion Lineal Multiple.** El techo de R² ≈ 0.32 no es un defecto
del modelo sino del problema: la calidad es una puntuacion **subjetiva y discreta** asignada
por catadores, y 11 propiedades quimicas no pueden explicar por completo una valoracion
sensorial."""),
]

# =============================================================================
# CASO 3
# =============================================================================
C3 = [
 ("md", CABECERA.format(
     titulo="Caso 3 — Diabetes: progresion de la enfermedad",
     contexto="**Contexto.** Predecir la progresion cuantitativa de la enfermedad un ano "
              "despues del inicio, a partir de 10 variables basales de 442 pacientes "
              "(edad, sexo, IMC, presion arterial y seis mediciones de suero sanguineo).")),
 ("code", IMPORTS),
 ("md", """## Fase A — Analisis exploratorio y preprocesamiento

### Carga: por que `scaled=False`

`load_diabetes()` devuelve por defecto los datos **ya centrados y escalados**, con valores
adimensionales entre −0.14 y 0.20. Se usa `scaled=False` de forma deliberada por dos razones:

1. La Fase A exige aplicar `StandardScaler`; no tiene sentido escalar lo ya escalado.
2. El aplicativo web necesita unidades clinicas reales: un usuario introduce
   "IMC = 26.4", no "IMC = 0.0507"."""),
 ("code", '''from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True, as_frame=True, scaled=False)
# 'sex' es la unica categorica y viene codificada 1/2: se recodifica a 0/1 para
# que su coeficiente se lea como el efecto de una categoria respecto a la otra.
X["sex"] = (X["sex"] - 1).astype(float)
print(f"Pacientes: {X.shape[0]}   Variables: {X.shape[1]}")
display(X.describe().T[["mean","std","min","50%","max"]].round(3))
print(f"Objetivo: rango {y.min():.0f} – {y.max():.0f}, media {y.mean():.2f}, "
      f"asimetria {y.skew():.3f}")'''),
 ("md", "**Significado de las variables.** `s1` colesterol total, `s2` LDL, `s3` HDL, "
        "`s4` razon colesterol/HDL, `s5` trigliceridos (posiblemente en log), `s6` glucosa. "
        "La documentacion de scikit-learn advierte que el significado exacto *\"puede no ser "
        "claro, especialmente para ltg\"*, asi que se reporta con esa reserva."),
 ("code", '''print(f"Nulos: {X.isna().sum().sum()}   Duplicados: {X.duplicated().sum()}")
print("No se requiere imputacion.")
print(f"\\nAsimetria del objetivo: {y.skew():.3f} -> aceptablemente simetrico,")
print("no hace falta transformacion logaritmica.")'''),
 ("md", "### Deteccion de outliers (metodo IQR)"),
 ("code", '''resumen = []
for c in X.columns:
    q1, q3 = X[c].quantile([.25, .75]); iqr = q3 - q1
    n = int(((X[c] < q1-1.5*iqr) | (X[c] > q3+1.5*iqr)).sum())
    resumen.append({"variable": c, "outliers": n, "%": round(100*n/len(X), 2)})
display(pd.DataFrame(resumen).set_index("variable"))

q1, q3 = y.quantile([.25, .75]); iqr = q3 - q1
print(f"Outliers en el objetivo: {int(((y < q1-1.5*iqr) | (y > q3+1.5*iqr)).sum())}")'''),
 ("md", "**Decision: se conservan.** Son valores clinicos plausibles (un colesterol de "
        "301 mg/dL es alto pero real) y con 442 muestras cada fila cuenta. Mas abajo se "
        "cuantifica el efecto de tratarlos."),
 ("md", MD_VIF),
 ("code", FUNC_VIF),
 ("code", '''corr = X.corr()
fig, ax = plt.subplots(figsize=(8.5, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True,
            linewidths=.5, ax=ax, cbar_kws={"shrink": .8})
ax.set_title("Matriz de correlacion de Pearson"); plt.tight_layout(); plt.show()

vif = calcular_vif(X)
display(pd.DataFrame({"VIF": vif.round(2), "diagnostico": vif.map(diagnostico)}))'''),
 ("md", """**El hallazgo central de este caso.** La multicolinealidad no es casual, es
**algebraica**:

- `s4` = colesterol total / HDL = **`s1` / `s3`** por definicion.
- `s2` (LDL) es un **componente** de `s1` (colesterol total).

Las variables lipidicas se contienen unas a otras. La consecuencia se vera en la Fase B:
sus errores estandar se inflan y sus coeficientes dejan de ser interpretables por separado."""),
 ("md", MD_FASE_B),
 ("code", FUNC_EVAL),
 ("code", '''X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, random_state=RANDOM_STATE)
print(f"Entrenamiento: {len(X_tr)}   Prueba: {len(X_te)}   (75 / 25)")
print("\\nAviso metodologico: con solo 111 muestras de prueba el R2 de test es ruidoso,")
print("por eso toda comparacion se hace sobre la validacion cruzada 5-fold.")'''),
 ("md", "### Fase B.1 — Regresion Lineal Multiple"),
 ("code", '''m_lin, pipe_lin = evaluar("Lineal multiple (OLS)", 1, X_tr, X_te, y_tr, y_te)
pd.Series(m_lin).to_frame("valor")'''),
 ("md", "#### Significancia estadistica de los coeficientes\n\n"
        "Prueba *t* sobre los coeficientes estandarizados. Es aqui donde se **mide** el "
        "efecto de la multicolinealidad detectada en la Fase A."),
 ("code", '''from scipy import stats

sc = StandardScaler().fit(X_tr)
A = np.column_stack([np.ones(len(X_tr)), sc.transform(X_tr)])
beta = np.linalg.lstsq(A, y_tr, rcond=None)[0]
resid = y_tr - A @ beta
gl = len(A) - A.shape[1]
cov = (resid @ resid / gl) * np.linalg.inv(A.T @ A)
se = np.sqrt(np.diag(cov))
t = beta / se
p = 2 * (1 - stats.t.cdf(np.abs(t), gl))

tabla = pd.DataFrame({"coef": beta.round(4), "error_std": se.round(4),
                      "t": t.round(2), "p_valor": p.round(4)},
                     index=["intercepto"] + list(X.columns))
tabla["signif"] = ["***" if v < .001 else "**" if v < .01 else "*" if v < .05 else ""
                   for v in p]
display(tabla)
print("Significativas al 5 %: " + ", ".join(v for v, q in zip(X.columns, p[1:]) if q < .05))'''),
 ("md", "Observa los **errores estandar**: `s1` tiene 22.21 y `s2` 17.98, frente a 3.3–3.7 de "
        "`age`, `sex` o `bmi`. Son entre 5 y 7 veces mayores, y son justamente las variables de "
        "VIF mas alto. Eso es exactamente lo que hace la multicolinealidad: **inflar la varianza "
        "de los estimadores**. El modelo sigue siendo valido para *predecir*, pero no para "
        "*explicar* el efecto aislado de cada lipido."),
 ("md", "### Fase B.2 — Regresion Polinomial\n\nSe exploran los grados 1 a 3, con y sin "
        "regularizacion Ridge, para cuantificar el sobreajuste."),
 ("code", '''from sklearn.linear_model import RidgeCV

filas = []
for grado in (1, 2, 3):
    m, _ = evaluar(f"Grado {grado} — OLS", grado, X_tr, X_te, y_tr, y_te)
    filas.append(m)
    pasos = [("poly", PolynomialFeatures(degree=grado, include_bias=False)),
             ("sc", StandardScaler()),
             ("modelo", RidgeCV(alphas=np.logspace(-2, 5, 71), cv=CV))]
    pipe = Pipeline(pasos).fit(X_tr, y_tr)
    cv = cross_val_score(pipe, X_tr, y_tr, cv=CV, scoring="r2")
    filas.append({"modelo": f"Grado {grado} — Ridge (a={pipe.named_steps['modelo'].alpha_:.2f})",
                  "terminos": pipe.named_steps["modelo"].coef_.size,
                  "R2_train": round(r2_score(y_tr, pipe.predict(X_tr)), 4),
                  "R2_test": round(r2_score(y_te, pipe.predict(X_te)), 4),
                  "R2_CV": round(cv.mean(), 4), "CV_std": round(cv.std(), 4),
                  "brecha": round(r2_score(y_tr, pipe.predict(X_tr)) - cv.mean(), 4),
                  "RMSE": round(np.sqrt(mean_squared_error(y_te, pipe.predict(X_te))), 4),
                  "MAE": round(mean_absolute_error(y_te, pipe.predict(X_te)), 4)})

pd.DataFrame(filas).set_index("modelo")'''),
 ("md", "### Fase B.3 — Comparativa y curva de validacion"),
 ("code", '''res = pd.DataFrame(filas)
ols = res[res.modelo.str.contains("OLS")]
rid = res[res.modelo.str.contains("Ridge")]
g = [1, 2, 3]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(g, ols.R2_train, "o-", c="#0d3c6c", label="OLS — entrenamiento")
ax.plot(g, ols.R2_CV, "o--", c="#0d3c6c", alpha=.65, label="OLS — validacion cruzada")
ax.plot(g, rid.R2_train, "s-", c="#1d9e6f", label="Ridge — entrenamiento")
ax.plot(g, rid.R2_CV, "s--", c="#1d9e6f", alpha=.65, label="Ridge — validacion cruzada")
ax.set_ylim(-.35, 1); ax.axhline(0, c="k", lw=.8)
ax.set_xticks(g); ax.set_xlabel("Grado del polinomio"); ax.set_ylabel("R2")
ax.set_title("Fase B — El sobreajuste crece con el grado")
for gr, v in zip(g, ols.R2_CV):
    if v < -.35:
        ax.annotate(f"R2 = {v:.1f}\\n(fuera de escala)", (gr, -.32), xytext=(-14, 8),
                    textcoords="offset points", ha="right", fontsize=9, color="#c0392b",
                    fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="#c0392b"))
ax.legend(loc="lower left", fontsize=9); plt.tight_layout(); plt.show()'''),
 ("md", """### Conclusion del Caso 3

El polinomio de grado 3 sin regularizar alcanza el **mejor R² de entrenamiento de todo el
estudio (0.8960)** y es, con enorme diferencia, **el peor modelo fuera de la muestra**
(R² de prueba −8.09, validacion cruzada −171.12). Un R² negativo significa que predice
**peor que responder siempre la media**. Es el argumento definitivo contra evaluar un
modelo por su ajuste en entrenamiento.

Con Ridge el sobreajuste desaparece en los tres grados, pero el resultado **converge al
del modelo lineal** (R² de CV ≈ 0.460): la penalizacion no extrae informacion de los
terminos polinomiales, los neutraliza.

La diferencia entre el mejor lineal (0.4606) y el mejor polinomial (0.4611) es de
**+0.0005**, frente a una desviacion entre particiones de ±0.083. Son **estadisticamente
indistinguibles**, y por parsimonia se recomienda la **Regresion Lineal Multiple**: mismo
rendimiento con 10 terminos en lugar de 65 y coeficientes interpretables."""),
]

if __name__ == "__main__":
    print("Generando cuadernos Jupyter…")
    cuaderno(C1, "CASO_1_analisis.ipynb", "CASO_1")
    cuaderno(C2, "CASO_2_analisis.ipynb", "CASO_2")
    cuaderno(C3, "CASO_3_analisis.ipynb", "CASO_3")
    print("\nListo.")
