html = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Wine Quality Predictor — Regresión Lineal Multiple vs Polinómica | Taller 1</title>
<meta name="description" content="Comparador de regresión lineal múltiple y polinómica para calidad de vino tinto."/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet"/>
<style>
:root{
  --bg:#0f0a12;--surface:#1a1025;--border:rgba(180,100,220,0.15);
  --lin:#3b82f6;--lin-g:rgba(59,130,246,0.3);--lin-d:#1d4ed8;
  --poly:#a855f7;--poly-g:rgba(168,85,247,0.3);--poly-d:#7e22ce;
  --gold:#f0c97a;--cream:#faf5ff;--ok:#34d399;--warn:#fb923c;
  --text:#e8d5f5;--text2:#b094cc;--text3:#6b4d8a;
  --r:10px;--r2:18px;--r3:26px;
}
*{box-sizing:border-box;margin:0;padding:0;}
html{scroll-behavior:smooth;}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden;}
body::before{content:'';position:fixed;inset:0;z-index:0;
  background:
    radial-gradient(ellipse 55% 45% at 15% 25%,rgba(59,130,246,.1) 0%,transparent 65%),
    radial-gradient(ellipse 45% 55% at 85% 75%,rgba(168,85,247,.12) 0%,transparent 65%),
    radial-gradient(ellipse 70% 35% at 50% 5%,rgba(20,10,40,.9) 0%,transparent 55%);
  pointer-events:none;}
.pts{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden;}
.pt{position:absolute;border-radius:50%;animation:flt linear infinite;}
@keyframes flt{0%{transform:translateY(105vh) scale(0);opacity:0}8%{opacity:.8}92%{opacity:.2}100%{transform:translateY(-5vh) scale(1.3);opacity:0}}
.wrap{position:relative;z-index:1;max-width:1380px;margin:0 auto;padding:0 18px 80px;}

/* HEADER */
header{text-align:center;padding:50px 0 32px;}
.badge{display:inline-flex;align-items:center;gap:8px;background:rgba(168,85,247,.15);border:1px solid rgba(168,85,247,.3);border-radius:100px;padding:5px 16px;font-size:11px;font-weight:700;letter-spacing:1.6px;text-transform:uppercase;color:#c084fc;margin-bottom:16px;}
h1{font-size:clamp(1.7rem,4.5vw,3rem);font-weight:800;line-height:1.1;
  background:linear-gradient(135deg,#e0d0ff 0%,#a78bfa 40%,#60a5fa 80%,#f0c97a 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:10px;}
.sub{font-size:.92rem;color:var(--text2);max-width:620px;margin:0 auto 24px;line-height:1.7;}

/* TOGGLE BAR */
.tbar{display:flex;align-items:center;justify-content:center;gap:0;background:rgba(255,255,255,.04);border:1px solid var(--border);border-radius:100px;padding:4px;width:fit-content;margin:0 auto 26px;}
.tbtn{padding:8px 24px;border-radius:100px;border:none;cursor:pointer;font-family:'Inter',sans-serif;font-size:13px;font-weight:600;transition:all .25s;background:transparent;color:var(--text2);}
.tbtn.lin.active{background:linear-gradient(135deg,var(--lin-d),var(--lin));color:#fff;box-shadow:0 2px 16px var(--lin-g);}
.tbtn.poly.active{background:linear-gradient(135deg,var(--poly-d),var(--poly));color:#fff;box-shadow:0 2px 16px var(--poly-g);}
.tbtn.both.active{background:linear-gradient(135deg,var(--lin-d),var(--poly));color:#fff;box-shadow:0 2px 18px rgba(168,85,247,.4);}
.tbtn:hover:not(.active){background:rgba(255,255,255,.06);color:var(--text);}

/* METRIC PILLS */
.mprow{display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin-bottom:26px;}
.mp{display:flex;align-items:center;gap:5px;background:rgba(255,255,255,.04);border:1px solid var(--border);border-radius:100px;padding:6px 14px;font-size:11.5px;}
.mp .lbl{color:var(--text3);}
.mp .val{font-weight:700;}
.mp.ml .val{color:#60a5fa;} .mp.mp2 .val{color:#c084fc;}

/* LAYOUT */
.main{display:grid;grid-template-columns:1fr 420px;gap:18px;align-items:start;}
@media(max-width:960px){.main{grid-template-columns:1fr;}}

/* CARD STYLES */
.card{background:rgba(255,255,255,.03);border:1px solid var(--border);border-radius:var(--r3);backdrop-filter:blur(14px);padding:22px;}
.ct{font-size:11px;font-weight:700;letter-spacing:1.3px;text-transform:uppercase;margin-bottom:16px;display:flex;align-items:center;gap:8px;}
.ct::before{content:'';width:16px;height:2px;border-radius:2px;}
.ct.cl{color:#60a5fa;} .ct.cl::before{background:linear-gradient(90deg,var(--lin-d),var(--lin));}
.ct.cp{color:#c084fc;} .ct.cp::before{background:linear-gradient(90deg,var(--poly-d),var(--poly));}
.ct.cm{color:#a78bfa;} .ct.cm::before{background:linear-gradient(90deg,var(--lin-d),var(--poly));}

/* FORM ELEMENTS */
.fgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px;}
.field{display:flex;flex-direction:column;gap:4px;}
.field label{font-size:10.5px;font-weight:600;text-transform:uppercase;letter-spacing:.7px;color:var(--text2);display:flex;justify-content:space-between;align-items:center;}
.fu{font-size:9px;color:var(--text3);font-weight:400;text-transform:none;letter-spacing:0;}
.field input[type=number]{width:100%;background:rgba(0,0,0,.4);border:1px solid rgba(180,120,240,.12);border-radius:var(--r);padding:8px 11px;color:var(--cream);font-family:'JetBrains Mono',monospace;font-size:13px;transition:border-color .2s,box-shadow .2s;-moz-appearance:textfield;}
.field input[type=number]::-webkit-inner-spin-button{opacity:.3;}
.field input:focus{outline:none;border-color:#a78bfa;box-shadow:0 0 0 3px rgba(167,139,250,.1);}
input[type=range]{width:100%;-webkit-appearance:none;height:3px;background:rgba(180,120,240,.12);border-radius:3px;cursor:pointer;margin-top:2px;}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:12px;height:12px;border-radius:50%;background:#a78bfa;box-shadow:0 0 6px rgba(167,139,250,.5);transition:transform .15s;}
input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.3);}
.fh{font-size:9px;color:var(--text3);}

/* PREDICT BUTTON */
.pbtn{width:100%;margin-top:18px;border:none;border-radius:var(--r2);padding:14px;color:#fff;font-family:'Inter',sans-serif;font-size:14.5px;font-weight:700;cursor:pointer;transition:transform .15s,box-shadow .15s;}
.pbtn.pl{background:linear-gradient(135deg,var(--lin-d),var(--lin));box-shadow:0 4px 18px var(--lin-g);}
.pbtn.pp{background:linear-gradient(135deg,var(--poly-d),var(--poly));box-shadow:0 4px 18px var(--poly-g);}
.pbtn.pb{background:linear-gradient(135deg,var(--lin-d),var(--poly));box-shadow:0 4px 18px rgba(168,85,247,.35);}
.pbtn:hover{transform:translateY(-2px);} .pbtn:active{transform:translateY(0);}

/* RIGHT PANEL */
.rp{display:flex;flex-direction:column;gap:14px;}
.rw{display:grid;grid-template-columns:1fr;gap:10px;transition:all .3s;}
.rw.bm{grid-template-columns:1fr 1fr;}
@media(max-width:600px){.rw.bm{grid-template-columns:1fr;}}
.rc{border-radius:var(--r3);padding:20px;text-align:center;backdrop-filter:blur(14px);display:flex;flex-direction:column;align-items:center;gap:9px;min-height:230px;justify-content:center;}
.rc.rl{background:linear-gradient(135deg,rgba(29,78,216,.18),rgba(59,130,246,.06));border:1px solid rgba(96,165,250,.22);}
.rc.rp2{background:linear-gradient(135deg,rgba(126,34,206,.18),rgba(168,85,247,.06));border:1px solid rgba(192,132,252,.22);}
.rem{color:var(--text3);font-size:12.5px;line-height:1.6;}
.remico{font-size:36px;margin-bottom:4px;display:block;}

/* GAUGE DISPLAYS */
.gw{width:155px;height:85px;}
.gsvg{width:155px;height:85px;}
.gbg{fill:none;stroke:rgba(255,255,255,.06);stroke-width:13;stroke-linecap:round;}
.gf{fill:none;stroke-width:13;stroke-linecap:round;stroke-dasharray:244;stroke-dashoffset:244;transition:stroke-dashoffset 1s cubic-bezier(.4,0,.2,1),stroke .5s;}
.rlbl{font-size:.95rem;font-weight:600;letter-spacing:.3px;}
.rl{color:#60a5fa;} .rp2c{color:#c084fc;}
.rmeta{font-size:9.5px;color:var(--text3);}
.qbt{width:100%;height:5px;background:rgba(255,255,255,.05);border-radius:100px;overflow:hidden;margin-top:3px;}
.qbf{height:100%;border-radius:100px;width:0%;transition:width 1.2s cubic-bezier(.4,0,.2,1);}
.qbfl{background:linear-gradient(90deg,var(--lin-d),var(--lin),#93c5fd);}
.qbfp{background:linear-gradient(90deg,var(--poly-d),var(--poly),#e9d5ff);}
.qbw{width:100%;}
.mtag{display:inline-flex;align-items:center;border-radius:100px;padding:2px 9px;font-size:9.5px;font-weight:700;letter-spacing:.5px;}
.mtagl{background:rgba(59,130,246,.15);color:#93c5fd;border:1px solid rgba(59,130,246,.22);}
.mtagp{background:rgba(168,85,247,.15);color:#d8b4fe;border:1px solid rgba(168,85,247,.22);}

/* COMPARISON TABLE */
.ct2{width:100%;border-collapse:collapse;font-size:11.5px;}
.ct2 th{text-align:left;padding:6px 7px;color:var(--text3);font-weight:600;text-transform:uppercase;letter-spacing:.6px;font-size:9.5px;border-bottom:1px solid rgba(255,255,255,.06);}
.ct2 td{padding:6px 7px;border-bottom:1px solid rgba(255,255,255,.04);}
.ct2 tr:last-child td{border-bottom:none;}
.ml2{color:#60a5fa;font-family:'JetBrains Mono',monospace;font-size:10.5px;}
.mp3{color:#c084fc;font-family:'JetBrains Mono',monospace;font-size:10.5px;}
.mlb{color:var(--text2);}
.wl{background:rgba(59,130,246,.08);border-radius:4px;padding:1px 5px;font-size:9px;color:#60a5fa;font-weight:700;border:1px solid rgba(59,130,246,.18);}
.wp{background:rgba(168,85,247,.08);border-radius:4px;padding:1px 5px;font-size:9px;color:#c084fc;font-weight:700;border:1px solid rgba(168,85,247,.18);}

/* COEFFICIENT BARS */
.coeflist{display:flex;flex-direction:column;}
.cr{display:flex;align-items:center;gap:7px;padding:5px 0;border-bottom:1px solid rgba(255,255,255,.04);font-size:11px;}
.cr:last-child{border-bottom:none;}
.cn{color:var(--text2);font-weight:500;flex:0 0 auto;max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.cb{flex:1;height:3px;background:rgba(255,255,255,.05);border-radius:2px;overflow:hidden;}
.cbf{height:100%;border-radius:2px;}
.cv{font-family:'JetBrains Mono',monospace;font-size:10px;flex:0 0 auto;}
.cv.pos{color:#34d399;} .cv.neg{color:#f87171;}

/* FORMULA BREAKDOWN */
.fc{background:rgba(255,255,255,.03);border:1px solid var(--border);border-radius:var(--r3);backdrop-filter:blur(14px);}
.fh2{padding:16px 20px 0;}
.fti{font-size:10.5px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;display:flex;align-items:center;gap:7px;}
.fti::before{content:'';width:13px;height:2px;border-radius:2px;}
.feq{font-family:'JetBrains Mono',monospace;font-size:9.5px;color:var(--gold);background:rgba(0,0,0,.35);border-radius:var(--r);padding:8px 12px;margin:8px 0;line-height:1.7;word-break:break-all;white-space:pre-wrap;}
.tlist{padding:0 20px 16px;max-height:320px;overflow-y:auto;}
.tlist::-webkit-scrollbar{width:3px;}
.tlist::-webkit-scrollbar-thumb{background:rgba(167,139,250,.25);border-radius:2px;}
.trow{display:flex;align-items:center;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(255,255,255,.04);gap:5px;font-size:10.5px;}
.trow:last-child{border-bottom:none;}
.tn{color:var(--text2);font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:145px;}
.tb{flex:1;height:3px;background:rgba(255,255,255,.04);border-radius:2px;min-width:18px;max-width:50px;overflow:hidden;}
.tbf{height:100%;border-radius:2px;}
.tv{font-family:'JetBrains Mono',monospace;font-size:9.5px;white-space:nowrap;}
.tv.pos{color:#34d399;} .tv.neg{color:#f87171;}

/* STATS BOXES */
.sg{display:grid;grid-template-columns:1fr 1fr;gap:7px;}
.sb{background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.06);border-radius:var(--r);padding:9px;text-align:center;}
.sv{font-size:1.2rem;font-weight:700;}.sl{font-size:9px;text-transform:uppercase;letter-spacing:.6px;color:var(--text3);margin-top:1px;}

/* INFO CHIPS */
.ics{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px;margin-top:16px;}
.ic{background:rgba(255,255,255,.03);border:1px solid var(--border);border-radius:var(--r2);padding:12px;}
.ii{font-size:18px;margin-bottom:4px;} .in{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:#c084fc;margin-bottom:2px;}
.id{font-size:10.5px;color:var(--text2);line-height:1.45;}
@keyframes fU{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
.fu{animation:fU .4s ease both;}
.hidden{display:none!important;}
.toast{position:fixed;bottom:22px;left:50%;transform:translateX(-50%) translateY(14px);background:linear-gradient(135deg,#4c1d95,#7c3aed);color:#fff;padding:10px 20px;border-radius:100px;font-size:13px;font-weight:600;box-shadow:0 6px 24px rgba(124,58,237,.5);opacity:0;transition:all .3s;z-index:999;white-space:nowrap;}
.toast.show{opacity:1;transform:translateX(-50%) translateY(0);}
.ovf-warn{margin-top:10px;padding:10px;background:rgba(251,146,60,.07);border:1px solid rgba(251,146,60,.18);border-radius:var(--r);font-size:10.5px;color:#fdba74;line-height:1.6;}
</style>
</head>
<body>
<div class="pts" id="pts"></div>
<div class="wrap">

<!-- HEADER SECTION -->
<header>
  <div class="badge">🍷 Taller 1 — Caso 2 | Machine Learning</div>
  <h1>Wine Quality Predictor</h1>
  <p class="sub">Comparador de modelos: <strong style="color:#60a5fa">Regresión Lineal Múltiple</strong> vs <strong style="color:#c084fc">Regresión Polinómica (grado 2)</strong><br>Entrenados con 1,143 muestras de vino tinto — WineQT Dataset</p>
  
  <!-- BOTONES DE MODO DE MODELO -->
  <div class="tbar">
    <button class="tbtn lin active" id="tLin" onclick="setMode('linear')">📈 Solo Lineal</button>
    <button class="tbtn both" id="tBoth" onclick="setMode('both')">⚖️ Comparar Ambos</button>
    <button class="tbtn poly" id="tPoly" onclick="setMode('poly')">📐 Solo Polinómico</button>
  </div>

  <!-- STRIP DE MÉTRICAS COMPARATIVAS -->
  <div class="mprow">
    <div class="mp ml"><span class="lbl">📈 R² Lineal</span><span class="val">0.3171</span></div>
    <div class="mp ml"><span class="lbl">RMSE</span><span class="val">0.6165</span></div>
    <div class="mp ml"><span class="lbl">Terms</span><span class="val">12</span></div>
    <div class="mp mp2"><span class="lbl">📐 R² Poly</span><span class="val">0.2809</span></div>
    <div class="mp mp2"><span class="lbl">RMSE</span><span class="val">0.6326</span></div>
    <div class="mp mp2"><span class="lbl" style="color:var(--gold)">Features</span><span class="val" style="color:var(--gold)">77+1</span></div>
  </div>
</header>

<!-- LAYOUT PRINCIPAL -->
<div class="main">
  <!-- COLUMNA IZQUIERDA: FORMULARIO DE ENTRADA -->
  <div>
    <div class="card">
      <div class="ct cl" id="fmtitle">Propiedades Físico-Químicas del Vino</div>
      <form id="wineForm">
        <div class="fgrid" id="fgrid"></div>
        <button type="submit" class="pbtn pl" id="pbtn">🔬 Predecir con Modelo Lineal</button>
      </form>
    </div>
    <div class="ics" id="ichips"></div>
    <div class="fc hidden" id="fc" style="margin-top:16px;">
      <div class="fh2">
        <div class="fti" id="fti" style="color:#60a5fa;">Desglose de la Fórmula</div>
        <div class="feq" id="feq"></div>
      </div>
      <div class="tlist" id="tlist"></div>
    </div>
  </div>

  <!-- COLUMNA DERECHA: PANELES DE RESULTADOS Y MÉTRICAS -->
  <div class="rp">
    <div class="rw" id="rw">
      <!-- RESULTADO MODELO LINEAL -->
      <div class="rc rl" id="rcL">
        <div class="rem" id="remL"><span class="remico">📈</span>Modelo Lineal<br><small>Presiona Predecir</small></div>
        <div class="hidden" id="rcLC" style="width:100%;display:flex;flex-direction:column;align-items:center;gap:99px;">
          <span class="mtag mtagl">📈 LINEAL OLS</span>
          <svg class="gsvg" viewBox="0 0 155 85">
            <path class="gbg" d="M14,75 A60,60,0,0,1,141,75"/>
            <path class="gf" id="gfL" d="M14,75 A60,60,0,0,1,141,75"/>
            <text x="77.5" y="65" text-anchor="middle" fill="#e0f2fe" font-size="23" font-weight="800" font-family="Inter,sans-serif" id="gnL">0</text>
            <text x="77.5" y="80" text-anchor="middle" fill="#5b8aaa" font-size="7.5" font-family="Inter,sans-serif">CALIDAD</text>
          </svg>
          <div class="rlbl rl" id="rLL">—</div>
          <div class="qbw"><div style="display:flex;justify-content:space-between;font-size:8.5px;color:var(--text3)"><span>0</span><span>10</span></div><div class="qbt"><div class="qbf qbfl" id="qbL"></div></div></div>
          <div class="rmeta" id="rmL"></div>
        </div>
      </div>

      <!-- RESULTADO MODELO POLINÓMICO -->
      <div class="rc rp2 hidden" id="rcP">
        <div class="rem" id="remP"><span class="remico">📐</span>Modelo Polinómico<br><small>Presiona Predecir</small></div>
        <div class="hidden" id="rcPC" style="width:100%;display:flex;flex-direction:column;align-items:center;gap:9px;">
          <span class="mtag mtagp">📐 POLY deg=2</span>
          <svg class="gsvg" viewBox="0 0 155 85">
            <path class="gbg" d="M14,75 A60,60,0,0,1,141,75"/>
            <path class="gf" id="gfP" d="M14,75 A60,60,0,0,1,141,75"/>
            <text x="77.5" y="65" text-anchor="middle" fill="#f3e8ff" font-size="23" font-weight="800" font-family="Inter,sans-serif" id="gnP">0</text>
            <text x="77.5" y="80" text-anchor="middle" fill="#8b6faa" font-size="7.5" font-family="Inter,sans-serif">CALIDAD</text>
          </svg>
          <div class="rlbl rp2c" id="rLP">—</div>
          <div class="qbw"><div style="display:flex;justify-content:space-between;font-size:8.5px;color:var(--text3)"><span>0</span><span>10</span></div><div class="qbt"><div class="qbf qbfp" id="qbP"></div></div></div>
          <div class="rmeta" id="rmP"></div>
        </div>
      </div>
    </div>

    <!-- TABLA COMPARATIVA DE PREDICCIONES -->
    <div class="card hidden" id="cmpCard">
      <div class="ct cm">Comparación de Predicciones</div>
      <table class="ct2">
        <thead><tr><th>Métrica</th><th style="color:#60a5fa">📈 Lineal</th><th style="color:#c084fc">📐 Poly</th><th>Ganador</th></tr></thead>
        <tbody id="cmpBody"></tbody>
      </table>
    </div>

    <!-- MÉTRICAS DE ENTRENAMIENTO -->
    <div class="card">
      <div class="ct cm">Métricas del Dataset WineQT</div>
      <div class="sg">
        <div class="sb"><div class="sv" style="color:#60a5fa">0.3171</div><div class="sl">R² Lineal Test</div></div>
        <div class="sb"><div class="sv" style="color:#c084fc">0.2809</div><div class="sl">R² Poly Test</div></div>
        <div class="sb"><div class="sv" style="color:#60a5fa">0.6165</div><div class="sl">RMSE Lineal</div></div>
        <div class="sb"><div class="sv" style="color:#c084fc">0.6326</div><div class="sl">RMSE Poly</div></div>
        <div class="sb"><div class="sv" style="color:#60a5fa">12</div><div class="sl">Terms Lineal</div></div>
        <div class="sb"><div class="sv" style="color:#c084fc">78</div><div class="sl">Terms Poly</div></div>
      </div>
      <div class="ovf-warn"><strong>Overfitting detectado:</strong> El modelo polinómico tiene R²_train=0.4591 vs R²_test=0.2809. Con 77 features sobre 1,143 muestras captura ruido. El modelo lineal es más robusto y generaliza mejor.</div>
    </div>

    <!-- COEFICIENTES BETA -->
    <div class="card">
      <div class="ct cl" id="coefTitle">Coeficientes Beta (estandarizados)</div>
      <div class="coeflist" id="coefList"></div>
    </div>
  </div>
</div>
</div>
<div class="toast" id="toast"></div>

<!-- 
=============================================================================
DOCUMENTACIÓN EXHAUSTIVA DE LÓGICA Y MATEMÁTICAS EN JAVASCRIPT
=============================================================================
-->
<script>
/**
 * ESTRUCTURA DE DATOS DEL MODELO 1: REGRESIÓN LINEAL MÚLTIPLE (OLS)
 * Exportado desde Python (wine_model_linear.json)
 * Ecuación: ŷ = β0 + Σ β_i * ((X_i - μ_i) / σ_i)
 */
const LIN = {
  intercept: 5.656455142231951, // Intercepto β0 (Calidad esperada para vino promedio)
  features: [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol"
  ],
  // Medias (μ) de cada variable en el conjunto de entrenamiento (X_train)
  means: {
    "fixed acidity": 8.258096280087534, "volatile acidity": 0.5310175054704594,
    "citric acid": 0.2657986870897156, "residual sugar": 2.519091903719909,
    "chlorides": 0.08647592997811804, "free sulfur dioxide": 15.706783369803064,
    "total sulfur dioxide": 45.838621444201316, "density": 0.9966827242888401,
    "pH": 3.314234135667395, "sulphates": 0.6556455142231953, "alcohol": 10.43918672501822
  },
  // Desviaciones estándar (σ) de cada variable en X_train
  stds: {
    "fixed acidity": 1.6954260412031144, "volatile acidity": 0.17894919917708096,
    "citric acid": 0.19487488821079296, "residual sugar": 1.3050102686115965,
    "chlorides": 0.04741058577816551, "free sulfur dioxide": 10.231880389516325,
    "total sulfur dioxide": 31.911931715986025, "density": 0.0019133479989592098,
    "pH": 0.15281284549872468, "sulphates": 0.16605497608869363, "alcohol": 1.0741877134665558
  },
  // Coeficientes de regresión (β_1 .. β_11) para variables estandarizadas Z-score
  coef: {
    "fixed acidity": 0.08704837017354405, "volatile acidity": -0.23912210345232315,
    "citric acid": -0.06608222229394528, "residual sugar": 0.005378510722629178,
    "chlorides": -0.08564866217749845, "free sulfur dioxide": 0.019243729876503263,
    "total sulfur dioxide": -0.0728567967560595, "density": -0.05865034648595574,
    "pH": -0.03808651485496864, "sulphates": 0.16157815006661094, "alcohol": 0.2863634785283069
  }
};

/**
 * ESTRUCTURA DE DATOS DEL MODELO 2: REGRESIÓN POLINÓMICA (GRADO 2)
 * Exportado desde Python (wine_model_poly.json)
 * Pipeline: StandardScaler -> PolynomialFeatures(degree=2, include_bias=False) -> LinearRegression
 * Posee 77 coeficientes correspondiente a las 77 variables expandidas.
 */
const POLY = {
  intercept: 5.754802195873957, // Intercepto β0 polinómico
  features: [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol"
  ],
  means: LIN.means,
  stds: LIN.stds,
  // Coeficientes β_k para cada uno de los 77 términos polinómicos expandidos
  coef: {
    "fixed acidity": 0.09194370988710356, "volatile acidity": -0.19906239080512825,
    "citric acid": -0.07946466609196945, "residual sugar": 0.05654242044961014,
    "chlorides": -0.001789732412472607, "free sulfur dioxide": 0.10002287869939885,
    "total sulfur dioxide": -0.1441791415107374, "density": -0.10123266637956477,
    "pH": -0.04858308163331418, "sulphates": 0.24648789996918172, "alcohol": 0.2919886944013289,
    "fixed acidity^2": 0.06477969336243167, "fixed acidity volatile acidity": -0.07061253642126337,
    "fixed acidity citric acid": -0.15734637758958492, "fixed acidity residual sugar": 0.035693411906689496,
    "fixed acidity chlorides": -0.00743363843830018, "fixed acidity free sulfur dioxide": -0.17162668119096589,
    "fixed acidity total sulfur dioxide": 0.150377795106346, "fixed acidity density": -0.1621098506139108,
    "fixed acidity pH": 0.03111963243107217, "fixed acidity sulphates": 0.10550161571335628,
    "fixed acidity alcohol": -0.041010777289298306, "volatile acidity^2": -0.011827575619583558,
    "volatile acidity citric acid": 0.0667184443688086, "volatile acidity residual sugar": -0.09233927898915774,
    "volatile acidity chlorides": -0.01706884658879715, "volatile acidity free sulfur dioxide": -0.034249182604286416,
    "volatile acidity total sulfur dioxide": 0.11761558238335201, "volatile acidity density": 0.05900505937891856,
    "volatile acidity pH": -0.0665890706503992, "volatile acidity sulphates": -0.0827591846445867,
    "volatile acidity alcohol": 0.06600208460527879, "citric acid^2": 0.03123578977675244,
    "citric acid residual sugar": -0.12457822658872578, "citric acid chlorides": -0.11829514921982397,
    "citric acid free sulfur dioxide": 0.09626741407035684, "citric acid total sulfur dioxide": -0.037697046925923156,
    "citric acid density": 0.16373117520234542, "citric acid pH": -0.21030157727004814,
    "citric acid sulphates": -0.02998950440666387, "citric acid alcohol": 0.1490335999120304,
    "residual sugar^2": -0.03252893402291615, "residual sugar chlorides": 0.10492459286585787,
    "residual sugar free sulfur dioxide": -0.06953216636925325, "residual sugar total sulfur dioxide": 0.1082404200890263,
    "residual sugar density": 0.06698965011729259, "residual sugar pH": -0.057986437018469256,
    "residual sugar sulphates": -0.04296553210842922, "residual sugar alcohol": 0.06994225521942257,
    "chlorides^2": 0.015930482017619277, "chlorides free sulfur dioxide": 0.013650635643071375,
    "chlorides total sulfur dioxide": -0.06671241455069525, "chlorides density": 0.06222131007142648,
    "chlorides pH": 0.003846465728573327, "chlorides sulphates": 0.04298524182007413,
    "chlorides alcohol": 0.03859854304051927, "free sulfur dioxide^2": -0.043208763109588516,
    "free sulfur dioxide total sulfur dioxide": -0.013672191317627189, "free sulfur dioxide density": 0.1758006551437425,
    "free sulfur dioxide pH": -0.03168767203064658, "free sulfur dioxide sulphates": -0.13152325141209548,
    "free sulfur dioxide alcohol": 0.16766263367019912, "total sulfur dioxide^2": 0.011300513403073156,
    "total sulfur dioxide density": -0.25247078850245824, "total sulfur dioxide pH": 0.07712296916814562,
    "total sulfur dioxide sulphates": 0.12142086856478175, "total sulfur dioxide alcohol": -0.20440469926056748,
    "density^2": 0.06017031065706921, "density pH": 0.04870075640954182,
    "density sulphates": -0.112670074378653, "density alcohol": -0.042636426511268394,
    "pH^2": -0.0238785315638307, "pH sulphates": 0.1036636002333373,
    "pH alcohol": 0.04706145615173661, "sulphates^2": -0.027405652829982002,
    "sulphates alcohol": 0.018844848726351116, "alcohol^2": -0.08935371392356031
  }
};

/**
 * DEFINICIÓN DE METADATOS Y LÍMITES DE CADA VARIABLE DEL FORMULARIO
 */
const FIELDS = [
  { k: "fixed acidity", l: "Acidez Fija", u: "g/L", mn: 4.6, mx: 15.9, st: .1, df: 7.9, ic: "🍋", d: "Ácido tartárico. Aporta frescura y estabilidad." },
  { k: "volatile acidity", l: "Acidez Volátil", u: "g/L", mn: .12, mx: 1.58, st: .01, df: .52, ic: "💨", d: "Ácido acético. Excesos = sabor avinagrado." },
  { k: "citric acid", l: "Ácido Cítrico", u: "g/L", mn: 0, mx: 1, st: .01, df: .26, ic: "🍊", d: "Añade frescura y notas frutales." },
  { k: "residual sugar", l: "Azúcar Residual", u: "g/L", mn: .9, mx: 15.5, st: .1, df: 2.2, ic: "🍬", d: "Azúcar post-fermentación. Define dulzura." },
  { k: "chlorides", l: "Cloruros", u: "g/L", mn: .012, mx: .611, st: .001, df: .079, ic: "🧂", d: "Sal. Excesos afectan negativamente el sabor." },
  { k: "free sulfur dioxide", l: "SO2 Libre", u: "mg/L", mn: 1, mx: 72, st: 1, df: 14, ic: "🛡️", d: "Conservante y antioxidante natural." },
  { k: "total sulfur dioxide", l: "SO2 Total", u: "mg/L", mn: 6, mx: 289, st: 1, df: 38, ic: "🧪", d: "SO2 total. Excesos = olor desagradable." },
  { k: "density", l: "Densidad", u: "g/mL", mn: .990, mx: 1.004, st: .0001, df: .9967, ic: "⚖️", d: "Relacionada con alcohol y azúcar." },
  { k: "pH", l: "pH", u: "0-14", mn: 2.74, mx: 4.01, st: .01, df: 3.31, ic: "🔬", d: "Acidez total. Vinos: pH 3.0-4.0." },
  { k: "sulphates", l: "Sulfatos", u: "g/L", mn: .33, mx: 2, st: .01, df: .66, ic: "💎", d: "Mejoran conservación y calidad." },
  { k: "alcohol", l: "Alcohol", u: "%vol", mn: 8.4, mx: 14.9, st: .1, df: 10.4, ic: "🥃", d: "Mayor alcohol tiende a mayor calidad." }
];

// Estado global para controlar qué modo está activo: 'linear', 'poly', o 'both'
let mode = 'linear';

/**
 * ---------------------------------------------------------------------------
 * ALGORITMO DE PREDICCIÓN 1: REGRESIÓN LINEAL MÚLTIPLE (OLS)
 * ---------------------------------------------------------------------------
 * Paso 1: Inicializar y = β0 (intercepto).
 * Paso 2: Para cada predictor i:
 *         a) Estandarizar valor ingresado: X_scaled_i = (X_i - μ_i) / σ_i
 *         b) Multiplicar por coeficiente: contribución_i = β_i * X_scaled_i
 *         c) Acumular en y: y += contribución_i
 * Paso 3: Retornar valor final predicho y array de contribuciones desglosadas.
 */
function predLin(v) {
  let y = LIN.intercept;
  const T = [{ n: "B0 Intercepto", c: LIN.intercept }];
  LIN.features.forEach(f => {
    const xs = (v[f] - LIN.means[f]) / LIN.stds[f]; // Z-score scaling
    const c = LIN.coef[f] * xs;                     // β_i * X_scaled_i
    y += c;
    T.push({ n: f, xi: v[f], mu: LIN.means[f], sg: LIN.stds[f], xs, b: LIN.coef[f], c });
  });
  return { y, T };
}

/**
 * ---------------------------------------------------------------------------
 * ALGORITMO DE PREDICCIÓN 2: REGRESIÓN POLINÓMICA (GRADO 2)
 * ---------------------------------------------------------------------------
 * Replicación exacta del comportamiento de scikit-learn (PolynomialFeatures deg=2):
 * Paso 1: Estandarizar las 11 variables de entrada a Z-score (Xsc_i).
 * Paso 2: Generar las 77 variables expandidas:
 *         a) Términos lineales (Grado 1): Xsc_1, Xsc_2, ..., Xsc_11
 *         b) Combinaciones de grado 2 (i <= j):
 *            - Cuadráticos (i == j): Xsc_i^2
 *            - Interacciones (i < j): Xsc_i * Xsc_j
 * Paso 3: Calcular suma ponderada: y = β0 + Σ (β_k * Feature_Expandida_k)
 */
function predPoly(v) {
  const F = POLY.features;
  
  // 1. Estandarizar entradas
  const Xsc = F.map(f => (v[f] - POLY.means[f]) / POLY.stds[f]);
  
  const EN = [], EV = [];
  
  // 2a. Expandir términos lineales (Grado 1)
  F.forEach((f, i) => { EN.push(f); EV.push(Xsc[i]); });
  
  // 2b. Expandir combinaciones cuadráticas e interacciones (Grado 2)
  for (let i = 0; i < F.length; i++) {
    for (let j = i; j < F.length; j++) {
      EN.push(i === j ? F[i] + "^2" : F[i] + " " + F[j]);
      EV.push(Xsc[i] * Xsc[j]);
    }
  }
  
  // 3. Evaluar modelo polinómico
  let y = POLY.intercept;
  const T = [{ n: "B0 Intercepto", c: POLY.intercept }];
  EN.forEach((n, k) => {
    const b = POLY.coef[n] || 0;
    const c = b * EV[k];
    y += c;
    T.push({ n, v: EV[k], b, c });
  });
  return { y, T };
}

/**
 * CLASIFICADOR Y CODIFICADOR DE COLORES PARA LA CALIDAD DEL VINO
 */
function qlbl(s) {
  const r = Math.round(s);
  if (r >= 8) return { t: "Excelente", col: "#f0c97a" };
  if (r >= 7) return { t: "Muy Bueno", col: "#34d399" };
  if (r >= 6) return { t: "Bueno", col: "#6ee7f7" };
  if (r >= 5) return { t: "Regular", col: "#c9a0b0" };
  if (r >= 4) return { t: "Bajo", col: "#fb923c" };
  return { t: "Deficiente", col: "#f87171" };
}

function gc(s) {
  return s >= 7 ? "#34d399" : s >= 5 ? "#a78bfa" : "#f87171";
}

/**
 * RENDERIZADOR DE RESULTADOS EN LA INTERFAZ GRÁFICA (GAUGE SVG + BARS)
 */
function showRes(y, T, side) {
  const cl = Math.min(10, Math.max(0, y)), p = cl / 10, lb = qlbl(y);
  const isL = side === 'lin';
  const remId = isL ? "remL" : "remP", rcId = isL ? "rcLC" : "rcPC";
  const gfId = isL ? "gfL" : "gfP", gnId = isL ? "gnL" : "gnP";
  const qbId = isL ? "qbL" : "qbP", rLId = isL ? "rLL" : "rLP", rmId = isL ? "rmL" : "rmP";
  
  document.getElementById(remId).classList.add("hidden");
  const rc = document.getElementById(rcId); rc.classList.remove("hidden"); rc.style.display = "flex";
  
  // Animar indicador del Gauge SVG cambiando el stroke-dashoffset
  document.getElementById(gfId).style.strokeDashoffset = 244 - (p * 244);
  document.getElementById(gfId).style.stroke = gc(cl);
  document.getElementById(gnId).textContent = cl.toFixed(2);
  
  // Actualizar barra lineal de 0 a 10
  document.getElementById(qbId).style.width = `${p * 100}%`;
  document.getElementById(rLId).textContent = lb.t;
  document.getElementById(rLId).style.color = lb.col;
  document.getElementById(rmId).textContent = `Pred: ${y.toFixed(4)} | Redondeado: ${Math.round(cl)}/10`;
}

/**
 * CONSTRUCTOR DE TABLA COMPARATIVA DE MODELOS
 */
function buildCmp(ly, py) {
  document.getElementById("cmpCard").classList.remove("hidden");
  const b = document.getElementById("cmpBody"); b.innerHTML = "";
  const lc = Math.min(10, Math.max(0, ly)), pc = Math.min(10, Math.max(0, py));
  const diff = Math.abs(ly - py);
  const rows = [
    { l: "Predicción cruda", lv: ly.toFixed(4), pv: py.toFixed(4), w: null },
    { l: "Calidad (0-10)", lv: lc.toFixed(2), pv: pc.toFixed(2), w: null },
    { l: "Diferencia entre modelos", lv: "—", pv: "±" + diff.toFixed(4), w: null },
    { l: "R² en Test (Mayor es mejor)", lv: "0.3171", pv: "0.2809", w: "lin" },
    { l: "RMSE en Test (Menor es mejor)", lv: "0.6165", pv: "0.6326", w: "lin" },
    { l: "N° de Parámetros/Features", lv: "12", pv: "78", w: null },
    { l: "Riesgo de Sobreajuste", lv: "Bajo", pv: "Alto", w: "lin" },
  ];
  rows.forEach(r => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td class="mlb">${r.l}</td><td class="ml2">${r.lv}</td><td class="mp3">${r.pv}</td><td>${r.w === "lin" ? '<span class="wl">Lineal</span>' : r.w === "poly" ? '<span class="wp">Poly</span>' : ''}</td>`;
    b.appendChild(tr);
  });
}

/**
 * RENDERIZADOR DE FÓRMULA MATEMÁTICA Y DESGLOSE PASO A PASO
 */
function showFormula(T, m) {
  const fc = document.getElementById("fc"); fc.classList.remove("hidden"); fc.classList.add("fu");
  const fti = document.getElementById("fti"), feq = document.getElementById("feq"), tl = document.getElementById("tlist");
  
  if (m === 'linear') {
    fti.textContent = "Desglose Lineal OLS — y = B0 + SUM(Bi * Xi_sc)";
    fti.style.color = "#60a5fa";
    feq.textContent = "y = B0 + B1*(X1-mu1)/s1 + B2*(X2-mu2)/s2 + ... + B11*(X11-mu11)/s11\n\ny_hat = " + T.reduce((a, t) => a + t.c, 0).toFixed(6);
  } else {
    fti.textContent = "Desglose Polinómico — Pipeline: Scale -> PolyExpand(77) -> LinReg";
    fti.style.color = "#c084fc";
    feq.textContent = "Pipeline: StandardScaler -> PolynomialFeatures(deg=2,no_bias) -> LinearRegression\n77 features: 11 lineales + 55 interacciones + 11 cuadráticos\n\ny_hat = " + T.reduce((a, t) => a + t.c, 0).toFixed(6);
  }
  
  tl.innerHTML = "";
  const topT = m === 'linear' ? T : [T[0], ...T.slice(1).sort((a, b) => Math.abs(b.c) - Math.abs(a.c)).slice(0, 18)];
  const mx = Math.max(...topT.slice(1).map(t => Math.abs(t.c)));
  
  topT.forEach((t, i) => {
    const row = document.createElement("div"); row.className = "trow fu"; row.style.animationDelay = `${i * 22}ms`;
    const ip = t.c >= 0, p2 = mx > 0 ? Math.abs(t.c) / mx * 100 : 0;
    const col = m === 'linear' ? (ip ? "#3b82f6" : "#60a5fa") : (ip ? "#a855f7" : "#c084fc");
    if (i === 0) {
      row.innerHTML = `<span class="tn" style="color:var(--gold);font-weight:700">B0 Intercepto</span><div class="tb"><div class="tbf" style="width:100%;background:var(--gold)"></div></div><span class="tv pos">${t.c.toFixed(6)}</span>`;
    } else {
      row.innerHTML = `<span class="tn" title="${t.n}">${t.n}</span><div class="tb"><div class="tbf" style="width:${p2}%;background:${col}"></div></div><span class="tv ${ip ? 'pos' : 'neg'}">${ip ? "+" : ""}${t.c.toFixed(6)}</span>`;
    }
    tl.appendChild(row);
  });
  
  if (m === 'poly') {
    const n = document.createElement("div"); n.style.cssText = "font-size:9px;color:var(--text3);margin-top:6px;"; n.textContent = "Top 18 términos de 77 totales por magnitud de impacto."; tl.appendChild(n);
  }
  
  const sr = document.createElement("div"); sr.className = "trow";
  const tot = T.reduce((a, t) => a + t.c, 0);
  sr.innerHTML = `<span class="tn" style="color:var(--gold);font-weight:700">TOTAL y_hat</span><div class="tb"></div><span class="tv" style="color:var(--gold);font-weight:700">${tot.toFixed(6)}</span>`;
  tl.appendChild(sr);
}

/**
 * CONMUTADOR DE MODOS DE VISUALIZACIÓN ('linear', 'poly', 'both')
 */
function setMode(m) {
  mode = m;
  ['Lin', 'Both', 'Poly'].forEach(x => document.getElementById("t" + x).classList.remove('active'));
  document.getElementById("t" + (m === 'linear' ? "Lin" : m === 'poly' ? "Poly" : "Both")).classList.add('active');
  
  const pb = document.getElementById("pbtn"), rw = document.getElementById("rw");
  const rcL = document.getElementById("rcL"), rcP = document.getElementById("rcP");
  const cc = document.getElementById("cmpCard"), ct = document.getElementById("coefTitle"), ft = document.getElementById("fmtitle");
  
  ct.className = "ct"; ft.className = "ct";
  if (m === 'linear') {
    pb.className = "pbtn pl"; pb.textContent = "Predecir con Modelo Lineal";
    rw.className = "rw"; rcP.classList.add("hidden"); rcL.classList.remove("hidden");
    ct.classList.add("cl"); ct.textContent = "Coeficientes Beta — Modelo Lineal";
    ft.classList.add("cl"); cc.classList.add("hidden"); buildCoefs('linear');
  } else if (m === 'poly') {
    pb.className = "pbtn pp"; pb.textContent = "Predecir con Modelo Polinómico";
    rw.className = "rw"; rcL.classList.add("hidden"); rcP.classList.remove("hidden");
    ct.classList.add("cp"); ct.textContent = "Top Coeficientes — Poly (deg=2)";
    ft.classList.add("cp"); cc.classList.add("hidden"); buildCoefs('poly');
  } else {
    pb.className = "pbtn pb"; pb.textContent = "Predecir y Comparar Ambos Modelos";
    rw.className = "rw bm"; rcL.classList.remove("hidden"); rcP.classList.remove("hidden");
    ct.classList.add("cm"); ct.textContent = "Coeficientes Comparados";
    ft.classList.add("cm"); buildCoefs('both');
  }
}

/**
 * RENDERIZADOR DE BARRAS DE COEFICIENTES EN EL PANEL DERECHO
 */
function buildCoefs(m) {
  const l = document.getElementById("coefList"); l.innerHTML = "";
  if (m === 'linear' || m === 'both') {
    const mx = Math.max(...LIN.features.map(f => Math.abs(LIN.coef[f])));
    LIN.features.forEach(f => {
      const c = LIN.coef[f], p2 = Math.abs(c) / mx * 100;
      const fd = FIELDS.find(x => x.k === f);
      const row = document.createElement("div"); row.className = "cr";
      row.innerHTML = `<span class="cn" title="${f}">${fd ? fd.ic + " " + fd.l : f}</span><div class="cb"><div class="cbf" style="width:${p2}%;background:${c >= 0 ? "#3b82f6" : "#60a5fa"}"></div></div><span class="cv ${c >= 0 ? 'pos' : 'neg'}">${c >= 0 ? "+" : ""}${c.toFixed(4)}</span>`;
      if (m === 'both') {
        const pc = POLY.coef[f] || 0;
        row.innerHTML += `<span class="cv ${pc >= 0 ? 'pos' : 'neg'}" style="color:${pc >= 0 ? '#a855f7' : '#c084fc'}">${pc >= 0 ? "+" : ""}${pc.toFixed(4)}</span>`;
      }
      l.appendChild(row);
    });
    if (m === 'both') {
      const n = document.createElement("div");
      n.style.cssText = "font-size:9px;color:var(--text3);margin-top:6px;display:flex;gap:12px;";
      n.innerHTML = '<span style="color:#60a5fa">▌ Lineal</span><span style="color:#c084fc">▌ Polinómico (parte lineal)</span>';
      l.appendChild(n);
    }
  } else {
    const pc = Object.entries(POLY.coef).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 15);
    const mx = Math.max(...pc.map(([, v]) => Math.abs(v)));
    pc.forEach(([n, c]) => {
      const p2 = Math.abs(c) / mx * 100, row = document.createElement("div"); row.className = "cr";
      const isQ = n.includes("^2"), isI = n.split(" ").length > 1 && !isQ;
      row.innerHTML = `<span class="cn" title="${n}">${isQ ? "🔲" : isI ? "🔗" : "📌"} ${n}</span><div class="cb"><div class="cbf" style="width:${p2}%;background:${c >= 0 ? "#a855f7" : "#c084fc"}"></div></div><span class="cv ${c >= 0 ? 'pos' : 'neg'}">${c >= 0 ? "+" : ""}${c.toFixed(4)}</span>`;
      l.appendChild(row);
    });
    const n = document.createElement("div"); n.style.cssText = "font-size:9px;color:var(--text3);margin-top:6px;"; n.innerHTML = "Top 15 de 77 coefs. 🔲=cuadrático 🔗=interacción 📌=lineal"; l.appendChild(n);
  }
}

/**
 * GENERACIÓN DINÁMICA DE CAMPOS E INPUTS EN EL FORMULARIO
 */
function buildForm() {
  const g = document.getElementById("fgrid");
  FIELDS.forEach(f => {
    const id = f.k.replace(/ /g, "_");
    const d = document.createElement("div"); d.className = "field";
    d.innerHTML = `<label for="i_${id}">${f.l}<span class="fu">${f.u}</span></label><input type="number" id="i_${id}" name="${f.k}" min="${f.mn}" max="${f.mx}" step="${f.st}" value="${f.df}" required/><input type="range" id="r_${id}" min="${f.mn}" max="${f.mx}" step="${f.st}" value="${f.df}"/><span class="fh">Rango: ${f.mn}–${f.mx} | Media: ${LIN.means[f.k].toFixed(3)}</span>`;
    g.appendChild(d);
    
    // Sincronización entre input numérico y slider de rango
    const inp = document.getElementById(`i_${id}`), rng = document.getElementById(`r_${id}`);
    inp.addEventListener("input", () => rng.value = inp.value);
    rng.addEventListener("input", () => inp.value = rng.value);
  });
}

function buildChips() {
  const s = document.getElementById("ichips");
  FIELDS.forEach(f => {
    const c = document.createElement("div"); c.className = "ic";
    c.innerHTML = `<div class="ii">${f.ic}</div><div class="in">${f.l}</div><div class="id">${f.d}</div>`;
    s.appendChild(c);
  });
}

function showToast(msg) {
  const t = document.getElementById("toast"); t.textContent = msg; t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 3200);
}

/**
 * MANEJADOR EVENTO SUBMIT DEL FORMULARIO
 */
document.getElementById("wineForm").addEventListener("submit", e => {
  e.preventDefault();
  const v = {};
  FIELDS.forEach(f => { v[f.k] = parseFloat(document.getElementById(`i_${f.k.replace(/ /g, "_")}`).value); });
  
  document.getElementById("pbtn").textContent = "Calculando...";
  
  setTimeout(() => {
    if (mode === 'linear') {
      const { y, T } = predLin(v);
      showRes(y, T, 'lin'); showFormula(T, 'linear');
      showToast("Lineal: " + y.toFixed(2) + " / 10");
    } else if (mode === 'poly') {
      const { y, T } = predPoly(v);
      showRes(y, T, 'poly'); showFormula(T, 'poly');
      showToast("Polinómico: " + y.toFixed(2) + " / 10");
    } else {
      const { y: ly, T: lT } = predLin(v);
      const { y: py, T: pT } = predPoly(v);
      showRes(ly, lT, 'lin'); showRes(py, pT, 'poly');
      buildCmp(ly, py); showFormula(lT, 'linear');
      showToast("Lineal: " + ly.toFixed(2) + " | Poly: " + py.toFixed(2));
    }
    document.getElementById("pbtn").textContent =
      mode === 'linear' ? "Predecir con Modelo Lineal" :
      mode === 'poly' ? "Predecir con Modelo Polinómico" : "Predecir y Comparar Ambos Modelos";
  }, 280);
});

function initPts() {
  const c = document.getElementById("pts");
  for (let i = 0; i < 18; i++) {
    const p = document.createElement("div"); p.className = "pt";
    const sz = Math.random() * 5 + 1.5, bl = Math.random() > .5;
    p.style.cssText = `width:${sz}px;height:${sz}px;left:${Math.random() * 100}%;background:radial-gradient(circle,${bl ? "rgba(96,165,250,.6)" : "rgba(192,132,252,.6)"},transparent);animation-duration:${Math.random() * 13 + 7}s;animation-delay:${Math.random() * 11}s;`;
    c.appendChild(p);
  }
}

// Inicializar la aplicación web
buildForm(); buildChips(); buildCoefs('linear'); initPts(); setMode('linear');
</script>
</body>
</html>"""

with open("wine_predictor_v2.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Archivo HTML generado exitosamente.")
