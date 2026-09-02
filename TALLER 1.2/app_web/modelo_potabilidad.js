// Generado por train_model.py - NO editar a mano.
// Parametros reales del pipeline (imputer + scaler + regresion logistica)
// ajustado sobre el set de entrenamiento. Ver train_model.py, Fase V.
const MODELO_POTABILIDAD = {
  "caso": "Taller 1.2 - Potabilidad del Agua | Regresion Logistica",
  "features": [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
  ],
  "descripcion": {
    "ph": "pH del agua (0-14, ideal 6.5-8.5)",
    "Hardness": "Dureza - capacidad de precipitar jabon (mg/L)",
    "Solids": "Solidos disueltos totales (ppm)",
    "Chloramines": "Cloraminas - desinfectante residual (ppm)",
    "Sulfate": "Sulfatos disueltos (mg/L)",
    "Conductivity": "Conductividad electrica (uS/cm)",
    "Organic_carbon": "Carbono organico total (ppm)",
    "Trihalomethanes": "Trihalometanos - subproducto de cloracion (ug/L)",
    "Turbidity": "Turbidez - material en suspension (NTU)"
  },
  "rangos_observados": {
    "ph": [
      0.0,
      13.999999999999998
    ],
    "Hardness": [
      47.432,
      323.124
    ],
    "Solids": [
      320.942611274359,
      61227.19600771213
    ],
    "Chloramines": [
      0.3520000000000003,
      13.127000000000002
    ],
    "Sulfate": [
      129.00000000000003,
      481.0306423059972
    ],
    "Conductivity": [
      181.483753985146,
      753.3426195583046
    ],
    "Organic_carbon": [
      2.1999999999999886,
      28.30000000000001
    ],
    "Trihalomethanes": [
      0.7379999999999995,
      124.0
    ],
    "Turbidity": [
      1.45,
      6.739
    ]
  },
  "imputacion_mediana_train": {
    "ph": 7.035036795196814,
    "Hardness": 196.92806093779086,
    "Solids": 20866.335842223954,
    "Chloramines": 7.118161983426347,
    "Sulfate": 333.073545745888,
    "Conductivity": 424.94133573042495,
    "Organic_carbon": 14.214779561341032,
    "Trihalomethanes": 66.56570894421057,
    "Turbidity": 3.9696018473949692
  },
  "escalado_train": {
    "media": {
      "ph": 7.07995859136582,
      "Hardness": 196.52082664028,
      "Solids": 21888.067749063168,
      "Chloramines": 7.116362217117437,
      "Sulfate": 333.7697224474682,
      "Conductivity": 427.91585266249865,
      "Organic_carbon": 14.272734607893634,
      "Trihalomethanes": 66.16434879033093,
      "Turbidity": 3.9737826576914994
    },
    "desviacion": {
      "ph": 1.4695257941190254,
      "Hardness": 32.631483920889416,
      "Solids": 8758.43741061701,
      "Chloramines": 1.5989147575029456,
      "Sulfate": 36.229228351058666,
      "Conductivity": 80.9283562647498,
      "Organic_carbon": 3.2959272502517587,
      "Trihalomethanes": 15.792200488291687,
      "Turbidity": 0.7801933425529433
    }
  },
  "regresion_logistica": {
    "intercepto": -0.0007289857024894229,
    "coeficientes": {
      "ph": 0.025633913883333206,
      "Hardness": -0.0021286694642798803,
      "Solids": 0.05926248935414956,
      "Chloramines": 0.041254494161276266,
      "Sulfate": -0.013301112270649042,
      "Conductivity": 0.006084000779433002,
      "Organic_carbon": -0.028627716296217638,
      "Trihalomethanes": 0.00819386098334667,
      "Turbidity": 0.002418139953815675
    },
    "umbral_decision": 0.5,
    "clases": {
      "0": "No potable",
      "1": "Potable"
    }
  },
  "metricas_test": {
    "matriz_confusion": {
      "TN": 208,
      "FP": 192,
      "FN": 120,
      "TP": 136
    },
    "accuracy": 0.524390243902439,
    "precision": 0.4146341463414634,
    "recall": 0.53125,
    "f1_score": 0.4657534246575342,
    "auc_roc": 0.5474316406249999
  }
};
