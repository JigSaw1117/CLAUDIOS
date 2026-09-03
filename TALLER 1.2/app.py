"""
Taller 1.2 - Clasificacion Binaria con Regresion Logistica
Potabilidad del Agua - Aplicativo web interactivo (Fase V)

Ejecutar local:  streamlit run app.py
Requiere:        modelo_potabilidad.pkl y resultados_metricas.json
                 (generados por train_model.py)
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Potabilidad del Agua | UAC",
    page_icon="💧",
    layout="centered",
)


@st.cache_resource
def cargar_modelo():
    return joblib.load(BASE_DIR / "modelo_potabilidad.pkl")


@st.cache_data
def cargar_resultados():
    with open(BASE_DIR / "resultados_metricas.json", encoding="utf-8") as f:
        return json.load(f)


modelo = cargar_modelo()
resultados = cargar_resultados()

DESCRIPCION = {
    "ph": "pH del agua (0-14, ideal 6.5-8.5)",
    "Hardness": "Dureza — capacidad de precipitar jabón (mg/L)",
    "Solids": "Sólidos disueltos totales (ppm)",
    "Chloramines": "Cloraminas — desinfectante residual (ppm)",
    "Sulfate": "Sulfatos disueltos (mg/L)",
    "Conductivity": "Conductividad eléctrica (μS/cm)",
    "Organic_carbon": "Carbono orgánico total (ppm)",
    "Trihalomethanes": "Trihalometanos — subproducto de cloración (μg/L)",
    "Turbidity": "Turbidez — material en suspensión (NTU)",
}

RANGOS = resultados["fase_v"]["rangos_observados"]
MEDIANAS = resultados["fase_v"]["medianas"]
FEATURES = list(RANGOS.keys())

st.title("💧 Clasificación de Potabilidad del Agua")
st.caption(
    "Taller 1.2 · Regresión Logística · Universidad Andina del Cusco — "
    "Inteligencia Artificial (2026-II)"
)

tab_clasificar, tab_metricas = st.tabs(["🔮 Clasificar", "📊 Métricas del modelo"])

with tab_clasificar:
    st.markdown(
        "Ingresa los parámetros fisicoquímicos de una muestra de agua para "
        "estimar si es **potable** o **no potable**, según el modelo entrenado "
        "sobre el dataset [Water Potability](https://www.kaggle.com/datasets/adityakadiwal/water-potability)."
    )

    with st.form("form_clasificacion"):
        col1, col2 = st.columns(2)
        entradas = {}
        for i, feat in enumerate(FEATURES):
            col = col1 if i % 2 == 0 else col2
            lo, hi = RANGOS[feat]
            entradas[feat] = col.number_input(
                label=feat,
                min_value=0.0,
                value=round(MEDIANAS[feat], 2),
                help=f"{DESCRIPCION.get(feat, feat)} · rango observado en datos: [{lo:.1f}, {hi:.1f}]",
                format="%.3f",
            )
        enviado = st.form_submit_button("Clasificar potabilidad", type="primary", use_container_width=True)

    if enviado:
        X_nuevo = pd.DataFrame([entradas])[FEATURES]
        clase = modelo.predict(X_nuevo)[0]
        proba = modelo.predict_proba(X_nuevo)[0]
        prob_potable = proba[1]

        if clase == 1:
            st.success(f"✅ **Potable** — probabilidad estimada: {prob_potable:.1%}")
        else:
            st.error(f"⛔ **No potable** — probabilidad de ser potable: {prob_potable:.1%}")

        st.progress(float(prob_potable), text=f"P(potable) = {prob_potable:.1%}")
        st.caption(
            "La clasificación usa un umbral de 0.5 sobre la probabilidad de la clase "
            "'Potable' que entrega la regresión logística."
        )

with tab_metricas:
    m = resultados["fase_iv"]
    cm = m["matriz_confusion"]

    st.subheader("Desempeño sobre el conjunto de prueba")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{m['accuracy']:.1%}")
    c2.metric("Precision", f"{m['precision']:.1%}")
    c3.metric("Recall", f"{m['recall']:.1%}")
    c4.metric("F1-score", f"{m['f1_score']:.1%}")
    c5.metric("AUC-ROC", f"{m['auc_roc']:.3f}")

    st.subheader("Matriz de confusión")
    cm_df = pd.DataFrame(
        [[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]],
        index=["Real: No potable", "Real: Potable"],
        columns=["Pred: No potable", "Pred: Potable"],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("Coeficientes del modelo (escala estandarizada)")
    coefs_df = pd.Series(resultados["fase_iii"]["coeficientes"], name="Coeficiente").sort_values(
        key=abs, ascending=False
    )
    st.bar_chart(coefs_df)
    st.caption(
        "Un coeficiente positivo aumenta la probabilidad estimada de que el agua "
        "sea potable; uno negativo la reduce. Magnitudes comparables porque las "
        "variables están estandarizadas."
    )

    st.info(
        f"Dataset: {resultados['fase_ii']['n_muestras']} muestras · "
        f"Entrenamiento: {resultados['fase_ii']['split']['train_size']} · "
        f"Prueba: {resultados['fase_ii']['split']['test_size']} · "
        "Modelo: Regresión Logística (class_weight='balanced')."
    )
