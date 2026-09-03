// Motor de inferencia GENERICO para la Regresion Logistica del Taller 1.2.
//
// No conoce "ph", "Solids" ni ningun coeficiente: todo -features,
// medianas de imputacion, media/desviacion del escalado, coeficientes,
// intercepto y umbral- viene del modelo entrenado (modelo_potabilidad.js),
// generado por train_model.py. Si se reentrena con otro dataset u otras
// variables, este archivo no cambia.

/**
 * Clasifica una muestra aplicando el mismo pipeline usado en el entrenamiento:
 * imputacion por mediana -> estandarizacion -> combinacion lineal -> sigmoide.
 *
 * @param {Object} valoresCrudos  { nombreFeature: numero | null | undefined }
 * @param {Object} modelo         MODELO_POTABILIDAD (ver modelo_potabilidad.js)
 * @returns {{ clase: number, etiqueta: string, probabilidad: number, detalle: Array }}
 */
function clasificar(valoresCrudos, modelo) {
  const { features, imputacion_mediana_train, escalado_train, regresion_logistica } = modelo;
  const { intercepto, coeficientes, umbral_decision, clases } = regresion_logistica;

  let z = intercepto;
  const detalle = [];

  for (const feature of features) {
    let x = valoresCrudos[feature];
    let imputado = false;

    if (x === null || x === undefined || x === "" || Number.isNaN(Number(x))) {
      x = imputacion_mediana_train[feature];
      imputado = true;
    } else {
      x = Number(x);
    }

    const media = escalado_train.media[feature];
    const desviacion = escalado_train.desviacion[feature];
    const xEstandarizado = (x - media) / desviacion;
    const coef = coeficientes[feature];
    const contribucion = coef * xEstandarizado;
    z += contribucion;

    detalle.push({ feature, valor: x, imputado, xEstandarizado, coeficiente: coef, contribucion });
  }

  const probabilidad = 1 / (1 + Math.exp(-z));
  const clase = probabilidad >= umbral_decision ? 1 : 0;

  return {
    clase,
    etiqueta: clases[String(clase)],
    probabilidad,
    z,
    detalle,
  };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { clasificar };
}
