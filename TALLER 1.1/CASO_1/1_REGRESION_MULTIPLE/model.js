// Generado por train.py - Caso 1 California Housing | Regresion Lineal Multiple
const MODELO_MULTIPLE = {
  "caso": "Caso 1 - California Housing | Regresion Lineal Multiple",
  "metodologia": "Ridge (L2, alpha por CV) sobre log(precio), 22 descriptores",
  "descriptores": [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
    "income_sq",
    "log_population",
    "log_households",
    "log_total_rooms",
    "dist_sf",
    "dist_la",
    "ocean_INLAND",
    "ocean_ISLAND",
    "ocean_NEAR BAY",
    "ocean_NEAR OCEAN"
  ],
  "numericas": [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
    "income_sq",
    "log_population",
    "log_households",
    "log_total_rooms",
    "dist_sf",
    "dist_la"
  ],
  "categorias": [
    "<1H OCEAN",
    "INLAND",
    "ISLAND",
    "NEAR BAY",
    "NEAR OCEAN"
  ],
  "categoria_base": "<1H OCEAN",
  "mu": [
    -119.5761405529954,
    35.66119476822987,
    28.40803740851179,
    2618.603212252643,
    538.8098400650583,
    1438.3774058010301,
    501.25284629981024,
    3.6769377270262944,
    5.366253060882419,
    0.21580162149931062,
    3.131640423022575,
    15.965175528417593,
    7.036572075091284,
    5.9898806883573545,
    7.623990019177686,
    3.425218097845027,
    2.476861121332758,
    0.3291542423420981,
    0.00020330712930333424,
    0.11012469503930604,
    0.1253727297370561
  ],
  "sigma": [
    2.011350124659912,
    2.1485818139125885,
    12.511107038141184,
    2162.6485903726993,
    421.13032773945054,
    1148.521496901273,
    383.32280200385156,
    1.5637469360444167,
    2.4010821343511726,
    0.06595553977108519,
    12.247822974963928,
    14.202485566816714,
    0.729788208933469,
    0.7207725050742285,
    0.7463546268606809,
    2.218592214937425,
    2.2539883929123925,
    0.46990608326589844,
    0.014257131391500488,
    0.31304511908318566,
    0.331141070203222
  ],
  "coef": [
    -0.1637220392836298,
    -0.11717323334811555,
    0.02404310641290823,
    -0.033854657358056386,
    0.034685848826941035,
    -0.013001782379231754,
    0.03263005470625876,
    0.4308258045298503,
    0.02231389367136746,
    0.03651683936070189,
    0.025043539776866793,
    -0.13949420196156476,
    -0.27369435841799394,
    0.24956169519121738,
    0.030639511415419072,
    -0.10285362331065959,
    -0.16162676436268986,
    -0.16275903640773612,
    0.008520289055162257,
    -0.009896837424959252,
    0.021657514066397977
  ],
  "intercept": 12.036376592365293,
  "bedrooms_median": 437.0,
  "log_target": true,
  "rangos": {
    "longitude": [
      -124.35,
      -114.31
    ],
    "latitude": [
      32.54,
      41.95
    ],
    "housing_median_age": [
      1.0,
      52.0
    ],
    "total_rooms": [
      2.0,
      39320.0
    ],
    "total_bedrooms": [
      2.0,
      6445.0
    ],
    "population": [
      3.0,
      35682.0
    ],
    "households": [
      2.0,
      6082.0
    ],
    "median_income": [
      0.4999,
      15.0001
    ]
  },
  "medianas": {
    "longitude": -118.5,
    "latitude": 34.27,
    "housing_median_age": 28.0,
    "total_rooms": 2111.0,
    "total_bedrooms": 436.0,
    "population": 1179.0,
    "households": 411.0,
    "median_income": 3.45
  },
  "target": {
    "min": 14999.0,
    "max": 500000.0,
    "media": 192477.92101651843
  },
  "metricas": {
    "nombre": "Ridge (L2, alpha por CV)",
    "n_terminos": 21,
    "r2_train": 0.6888755952238683,
    "r2_test": 0.693913136008267,
    "r2_cv": 0.6829866131476485,
    "r2_cv_std": 0.012844938077148193,
    "rmse_test": 56285.43909930486,
    "mae_test": 39546.29338024787,
    "brecha": 0.005888982076219795
  },
  "comparativa": [
    {
      "nombre": "OLS - 8 originales + dummies",
      "n_terminos": 12,
      "r2_train": 0.6517647698111737,
      "r2_test": 0.6599852890527067,
      "r2_cv": 0.6483673708114587,
      "r2_cv_std": 0.013922370008735276,
      "rmse_test": 66811.06432289771,
      "mae_test": 42913.0406022608,
      "brecha": 0.003397398999715029
    },
    {
      "nombre": "OLS - 22 descriptores",
      "n_terminos": 21,
      "r2_train": 0.6894516531820001,
      "r2_test": 0.6956006022206475,
      "r2_cv": 0.6820105262485876,
      "r2_cv_std": 0.014529037392134174,
      "rmse_test": 55961.820418378345,
      "mae_test": 39404.27662190598,
      "brecha": 0.0074411269334124786
    },
    {
      "nombre": "Ridge (L2, alpha por CV)",
      "n_terminos": 21,
      "r2_train": 0.6888755952238683,
      "r2_test": 0.693913136008267,
      "r2_cv": 0.6829866131476485,
      "r2_cv_std": 0.012844938077148193,
      "rmse_test": 56285.43909930486,
      "mae_test": 39546.29338024787,
      "brecha": 0.005888982076219795
    },
    {
      "nombre": "Lasso (L1, alpha por CV)",
      "n_terminos": 21,
      "r2_train": 0.6885547910103249,
      "r2_test": 0.6932827798660258,
      "r2_cv": 0.6839425848008341,
      "r2_cv_std": 0.011838170690790885,
      "rmse_test": 56430.553225122145,
      "mae_test": 39605.1617098397,
      "brecha": 0.004612206209490766
    }
  ],
  "vif": {
    "longitude": 40.67441012608766,
    "latitude": 33.390277714494225,
    "housing_median_age": 1.4455672340092642,
    "total_rooms": 19.218277405980732,
    "total_bedrooms": 36.87651275536549,
    "population": 13.256147606668533,
    "households": 47.90377341260945,
    "median_income": 18.003867141865374,
    "rooms_per_household": 3.752980195505241,
    "bedrooms_per_room": 3.146663183087432,
    "population_per_household": 1.1830732361511862,
    "income_sq": 14.921648341393281,
    "log_population": 19.33053304565552,
    "log_households": 85.56096471529958,
    "log_total_rooms": 68.59189867445549,
    "dist_sf": 16.253604050442583,
    "dist_la": 13.65478270545267,
    "ocean_INLAND": 3.7490116035904277,
    "ocean_ISLAND": 1.0028994770061266,
    "ocean_NEAR BAY": 1.9700662055083127,
    "ocean_NEAR OCEAN": 1.6183296889344654
  },
  "n_train": 14756,
  "n_test": 4919,
  "censurados_descartados": 965,
  "nulos_imputados": 207
};
