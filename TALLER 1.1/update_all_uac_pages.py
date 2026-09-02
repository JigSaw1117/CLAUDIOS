"""
Script Python para regenerar todas las páginas HTML con:
1. Logo oficial de la UAC (Cóndor arriba + Universidad Andina del Cusco).
2. Fondo con overlay del Campus UAC y el lema oficial: "No es mi hogar, pero lo conozco de memoria".
3. Tira multicolor oficial UAC.
4. Navegación entre portada y los 3 casos.
5. Fuentes grandes, legibles y de alto contraste (sin clichés de IA).
"""

import os

# LOGO SVG OFICIAL UAC CON CÓNDOR ARRIBA Y TEXTO
OFFICIAL_UAC_LOGO_HTML = r'''
<div class="uac-brand-box" style="display:inline-flex; align-items:center; gap:12px;">
  <svg viewBox="0 0 200 200" width="48" height="48" style="flex-shrink:0;">
    <g fill="#0b3c6d">
      <!-- Condor Head -->
      <path d="M 100 25 C 95 25 90 30 90 38 C 90 46 95 52 100 58 L 100 66 C 92 66 84 60 80 52 C 76 44 75 36 77 28 C 80 18 88 10 100 8 C 106 7 112 8 116 12 C 120 16 122 22 121 28 L 112 28 C 113 22 110 18 106 16 C 102 14 98 16 95 20 C 92 24 92 30 94 34 L 100 25 Z" fill="#00b4d8"/>
      <!-- Condor Collar & Neck -->
      <path d="M 100 45 L 110 58 L 100 78 L 90 58 Z" fill="#ffffff"/>
      <!-- Condor Left Wing -->
      <path d="M 88 70 C 65 45 35 15 15 5 C 13 3 8 5 10 9 L 25 35 C 18 25 5 12 0 8 C -2 6 -5 8 -3 12 L 12 42 C 5 34 -6 22 -10 18 C -12 16 -15 18 -13 21 L 4 55 C -2 48 -12 38 -16 34 C -18 32 -21 34 -19 37 L 4 75 C -8 60 -18 50 -22 45 M 88 70 L 40 105 L 15 100 L 45 80 Z"/>
      <!-- Wings Full Geometry -->
      <path d="M 90 65 L 20 20 L 45 50 L 10 30 L 35 60 L 5 45 L 30 75 L 15 70 L 40 95 L 75 110 L 95 100 Z"/>
      <path d="M 110 65 L 180 20 L 155 50 L 190 30 L 165 60 L 195 45 L 170 75 L 185 70 L 160 95 L 125 110 L 105 100 Z"/>
      <!-- Tail Feathers -->
      <path d="M 80 105 L 70 145 L 85 150 L 90 165 L 100 155 L 110 165 L 115 150 L 130 145 L 120 105 Z"/>
    </g>
  </svg>
  <div style="display:flex; flex-direction:column; text-align:left; line-height:1.15;">
    <span style="font-size:15px; font-weight:800; color:#ffffff; letter-spacing:0.3px;">Universidad</span>
    <span style="font-size:15px; font-weight:800; color:#00b4d8; letter-spacing:0.3px;">Andina</span>
    <span style="font-size:13px; font-weight:700; color:#cbd5e1; letter-spacing:0.3px;">del Cusco</span>
  </div>
</div>
'''

# LEATHER SLOGAN WITH COLOR STRIP
SLOGAN_HTML = r'''
<div class="uac-slogan-box" style="margin: 20px auto 30px; text-align: center;">
  <div style="font-size: 1.35rem; font-weight: 800; color: #ffffff; letter-spacing: -0.2px; text-shadow: 0 2px 10px rgba(0,0,0,0.5);">
    "No es mi hogar, pero lo conozco de memoria"
  </div>
  <div style="width: 140px; height: 4px; background: linear-gradient(90deg, #00b4d8 0%, #48cae4 25%, #ffd166 50%, #f77f00 75%, #d62828 100%); margin: 10px auto 0; border-radius: 4px;"></div>
</div>
'''

def get_root_index_html():
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Taller 1 — Inteligencia Artificial | Universidad Andina del Cusco</title>
  <meta name="description" content="Modelado Predictivo Multisectorial con Regresión Lineal Múltiple, Polinomial y Despliegue de Aplicativos de IA. Universidad Andina del Cusco." />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet" />
  <style>
    :root {{
      --uac-navy: #0d3c6c;
      --uac-dark: #0a2540;
      --uac-cyan: #00b4d8;
      --uac-blue: #0077b6;
      --uac-light-cyan: #e6f9f6;
      --uac-gold: #d97706;
      --text-primary: #0f172a;
      --text-muted: #475569;
      --card-bg: #ffffff;
      --radius: 16px;
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: #f8fafc;
      color: var(--text-primary);
      line-height: 1.6;
      font-size: 16px;
    }}

    .top-strip {{
      height: 5px;
      background: linear-gradient(90deg, #00b4d8 0%, #48cae4 25%, #ffd166 50%, #f77f00 75%, #d62828 100%);
    }}

    header.uac-nav {{
      background: var(--uac-dark);
      color: white;
      position: sticky; top: 0; z-index: 100;
      box-shadow: 0 4px 14px rgba(10,37,64,0.2);
    }}
    .nav-container {{
      max-width: 1240px; margin: 0 auto;
      padding: 12px 24px;
      display: flex; justify-content: space-between; align-items: center;
    }}

    nav ul {{
      display: flex; list-style: none; gap: 8px; align-items: center;
    }}
    nav ul li a {{
      color: #cbd5e1; text-decoration: none;
      font-weight: 600; font-size: 14px;
      padding: 8px 16px; border-radius: 8px;
      transition: all 0.2s ease;
    }}
    nav ul li a:hover {{
      background: rgba(255,255,255,0.1); color: #ffffff;
    }}
    nav ul li a.active {{
      background: var(--uac-cyan); color: var(--uac-dark);
      font-weight: 700;
    }}

    /* HERO CON OVERLAY Y CAMPUS DE FONDO */
    .hero {{
      background: url('uac_campus_hero_bg.svg') center/cover no-repeat, linear-gradient(135deg, var(--uac-dark) 0%, var(--uac-navy) 100%);
      color: white; padding: 60px 24px 80px;
      text-align: center;
      position: relative;
      border-bottom: 4px solid var(--uac-cyan);
    }}
    .hero-container {{ max-width: 1000px; margin: 0 auto; position: relative; z-index: 2; }}
    
    .academic-badge {{
      display: inline-block;
      background: rgba(0,180,216,0.22); border: 1px solid rgba(0,180,216,0.45);
      padding: 6px 18px; border-radius: 100px;
      font-size: 13px; font-weight: 800; letter-spacing: 1.2px; text-transform: uppercase;
      color: #38bdf8; margin-bottom: 16px;
    }}
    
    .hero h1 {{
      font-size: clamp(2.1rem, 4.2vw, 3.2rem); font-weight: 800; line-height: 1.18;
      margin-bottom: 14px; color: #ffffff; text-shadow: 0 2px 10px rgba(0,0,0,0.5);
    }}
    
    .hero p {{
      font-size: 1.12rem; color: #e2e8f0; max-width: 840px; margin: 0 auto 24px;
      line-height: 1.7; text-shadow: 0 1px 4px rgba(0,0,0,0.4);
    }}

    .meta-pills {{
      display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;
    }}
    .meta-pill {{
      background: rgba(10,37,64,0.75); border: 1px solid rgba(255,255,255,0.2);
      padding: 8px 18px; border-radius: 100px; font-size: 14px; color: #f8fafc;
      font-weight: 600; backdrop-filter: blur(4px);
    }}
    .meta-pill strong {{ color: #fbbf24; font-weight: 700; }}

    .container {{ max-width: 1180px; margin: -30px auto 60px; padding: 0 24px; position: relative; z-index: 3; }}

    .info-card {{
      background: var(--card-bg); border-radius: var(--radius);
      padding: 32px; box-shadow: 0 4px 20px rgba(10,37,64,0.06);
      margin-bottom: 36px; border: 1px solid #e2e8f0; border-top: 4px solid var(--uac-navy);
    }}
    .info-card h2 {{
      font-size: 1.35rem; color: var(--uac-navy); margin-bottom: 14px; font-weight: 800;
      display: flex; align-items: center; gap: 10px;
    }}
    .info-card p {{ color: var(--text-muted); font-size: 1.02rem; line-height: 1.75; }}

    .section-title {{ text-align: center; margin-bottom: 28px; }}
    .section-title h2 {{ font-size: 1.8rem; color: var(--uac-navy); font-weight: 800; }}
    .section-title p {{ color: var(--text-muted); font-size: 1rem; }}

    .cases-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 24px;
      margin-bottom: 44px;
    }}

    .case-card {{
      background: var(--card-bg); border-radius: var(--radius);
      overflow: hidden; box-shadow: 0 4px 16px rgba(10,37,64,0.05);
      border: 1px solid #e2e8f0; transition: transform 0.2s, box-shadow 0.2s;
      display: flex; flex-direction: column;
    }}
    .case-card:hover {{
      transform: translateY(-4px); box-shadow: 0 10px 25px rgba(10,37,64,0.1);
    }}

    .case-header {{ padding: 22px 24px; color: white; }}
    .case-card.c1 .case-header {{ background: linear-gradient(135deg, #0d3c6c, #0077b6); }}
    .case-card.c2 .case-header {{ background: linear-gradient(135deg, #4a1525, #8b1a35); }}
    .case-card.c3 .case-header {{ background: linear-gradient(135deg, #1e3a8a, #3b82f6); }}

    .case-tag {{
      font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: 1.2px; opacity: 0.9; margin-bottom: 4px;
    }}
    .case-title {{ font-size: 1.35rem; font-weight: 800; }}

    .case-body {{ padding: 24px; flex: 1; display: flex; flex-direction: column; justify-content: space-between; }}
    .case-desc {{ color: var(--text-muted); font-size: 0.96rem; margin-bottom: 18px; line-height: 1.6; }}
    
    .case-features {{
      list-style: none; margin-bottom: 22px; font-size: 0.9rem; color: #334155;
    }}
    .case-features li {{ padding: 6px 0; border-bottom: 1px solid #f1f5f9; display: flex; justify-content: space-between; }}
    .case-features li strong {{ color: var(--uac-navy); font-weight: 700; }}

    .status-badge {{
      display: inline-flex; align-items: center; gap: 6px;
      padding: 6px 14px; border-radius: 100px; font-size: 13px; font-weight: 700; margin-bottom: 16px;
    }}
    .status-badge.completed {{ background: #d1fae5; color: #065f46; }}
    .status-badge.pending {{ background: #fef3c7; color: #92400e; }}

    .case-btn {{
      display: inline-flex; align-items: center; justify-content: center; gap: 8px;
      padding: 13px 20px; border-radius: 10px; font-weight: 700; font-size: 15px;
      text-decoration: none; transition: all 0.2s ease; width: 100%;
    }}
    .case-btn.active-btn {{
      background: #8b1a35; color: white;
    }}
    .case-btn.active-btn:hover {{ background: #6b1428; }}
    .case-btn.sec-btn {{
      background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1;
    }}
    .case-btn.sec-btn:hover {{ background: #e2e8f0; color: #0f172a; }}

    .team-card {{
      background: var(--uac-light-cyan); border: 1px solid #b2f0e8;
      border-radius: var(--radius); padding: 32px; margin-bottom: 40px;
    }}
    .team-card h3 {{
      color: var(--uac-navy); font-size: 1.3rem; font-weight: 800; margin-bottom: 20px;
      display: flex; align-items: center; gap: 10px;
    }}
    .team-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 16px;
    }}
    .team-member {{
      background: white; border-radius: 12px; padding: 18px 20px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.03); border-left: 4px solid var(--uac-blue);
      display: flex; align-items: center; gap: 14px;
    }}
    .team-avatar {{
      width: 44px; height: 44px; border-radius: 50%; background: var(--uac-navy);
      color: white; font-weight: 800; font-size: 15px;
      display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    }}
    .team-info .name {{ font-weight: 700; font-size: 0.95rem; color: var(--uac-dark); }}
    .team-info .role {{ font-size: 0.8rem; color: var(--text-muted); font-weight: 500; }}

    footer {{
      background: var(--uac-dark); color: #94a3b8; padding: 36px 24px;
      text-align: center; font-size: 0.92rem;
    }}
    footer strong {{ color: white; }}
    footer .footer-sub {{ font-size: 0.84rem; margin-top: 6px; color: #64748b; }}
  </style>
</head>
<body>

  <div class="top-strip"></div>

  <!-- NAVBAR MENU -->
  <header class="uac-nav">
    <div class="nav-container">
      <a href="index.html" style="text-decoration:none;">
        {OFFICIAL_UAC_LOGO_HTML}
      </a>
      <nav>
        <ul>
          <li><a href="index.html" class="active">🏠 Portada</a></li>
          <li><a href="CASO_1/index.html">📌 Caso 1</a></li>
          <li><a href="CASO_2/index.html">🍷 Caso 2 (Vinos)</a></li>
          <li><a href="CASO_3/index.html">📈 Caso 3</a></li>
        </ul>
      </nav>
    </div>
  </header>

  <!-- HERO ACADÉMICO CON SLOGAN -->
  <section class="hero" id="inicio">
    <div class="hero-container">
      <div class="academic-badge">UNIVERSIDAD ANDINA DEL CUSCO — TALLER 1</div>
      
      {SLOGAN_HTML}

      <h1>Modelado Predictivo Multisectorial con Regresión Lineal Múltiple, Polinomial y Despliegue de Aplicativos de IA</h1>
      <p>Asignatura: <strong>Inteligencia Artificial / Aprendizaje Automático</strong> (Ingeniería de Sistemas) — Unidad I</p>
      
      <div class="meta-pills">
        <div class="meta-pill">Docente: <strong>Espetia Huamanga, Hugo</strong></div>
        <div class="meta-pill">Semestre: <strong>2026-II</strong></div>
        <div class="meta-pill">Modalidad: <strong>Trabajo Práctico Grupal</strong></div>
        <div class="meta-pill">Ubicación: <strong>Cusco – Perú</strong></div>
      </div>
    </div>
  </section>

  <!-- CONTENIDO PRINCIPAL -->
  <main class="container">

    <div class="info-card">
      <h2>🎯 Objetivo de la Actividad</h2>
      <p>
        Esta actividad práctica tiene como objetivo que cada equipo de desarrollo aborde el ciclo de vida completo de un proyecto de Inteligencia Artificial. Los equipos deberán analizar, entrenar, evaluar y comparar modelos predictivos basados en <strong>Regresión Lineal Múltiple</strong> y <strong>Regresión Polinomial</strong> aplicando los <strong>tres (3) casos de estudio descritos</strong>. El resultado final se integrará en un único aplicativo interactivo desplegado en la nube e incluirá un informe técnico consolidado.
      </p>
    </div>

    <div class="section-title">
      <h2>Matriz de Casos de Estudio</h2>
      <p>Selecciona un módulo para abrir la aplicación interactiva correspondiente</p>
    </div>

    <div class="cases-grid">
      
      <!-- CASO 1 -->
      <div class="case-card c1">
        <div class="case-header">
          <div class="case-tag">Caso de Estudio 01</div>
          <div class="case-title">Módulo Sectorial 1</div>
        </div>
        <div class="case-body">
          <div class="case-desc">
            Modelado predictivo sectorial en desarrollo por el equipo. Incluye evaluación mediante Regresión Lineal Múltiple y Polinómica.
          </div>
          <ul class="case-features">
            <li><span>Algoritmos:</span> <strong>Regresión Lineal & Poly</strong></li>
            <li><span>Estado:</span> <strong>En Desarrollo</strong></li>
          </ul>
          <div>
            <span class="status-badge pending">⏳ Pendiente de Subida</span>
            <a href="CASO_1/index.html" class="case-btn sec-btn">Acceder al Caso 1 ➡</a>
          </div>
        </div>
      </div>

      <!-- CASO 2 - VINOS -->
      <div class="case-card c2">
        <div class="case-header">
          <div class="case-tag">Caso de Estudio 02</div>
          <div class="case-title">Calidad del Vino Tinto</div>
        </div>
        <div class="case-body">
          <div class="case-desc">
            Predicción de la calidad del vino (escala 0–10) evaluando 11 propiedades químico-físicas (acidez, azúcar residual, cloruros, alcohol, etc.).
          </div>
          <ul class="case-features">
            <li><span>Dataset:</span> <strong>WineQT.csv (1,143 filas)</strong></li>
            <li><span>Modelo Lineal:</span> <strong>R² = 0.3171 | RMSE = 0.6165</strong></li>
            <li><span>Modelo Poly:</span> <strong>R² = 0.2809 (Overfitting)</strong></li>
            <li><span>Despliegue:</span> <strong>Cliente JS + 2 JSONs</strong></li>
          </ul>
          <div>
            <span class="status-badge completed">✅ 100% Completado</span>
            <a href="CASO_2/index.html" class="case-btn active-btn">🍷 Abrir Predictor de Vino ➡</a>
          </div>
        </div>
      </div>

      <!-- CASO 3 -->
      <div class="case-card c3">
        <div class="case-header">
          <div class="case-tag">Caso de Estudio 03</div>
          <div class="case-title">Módulo Sectorial 3</div>
        </div>
        <div class="case-body">
          <div class="case-desc">
            Tercer escenario práctico del taller. Integración de predictores multisectoriales y comparación de métricas de precisión.
          </div>
          <ul class="case-features">
            <li><span>Algoritmos:</span> <strong>Regresión Lineal & Poly</strong></li>
            <li><span>Estado:</span> <strong>En Desarrollo</strong></li>
          </ul>
          <div>
            <span class="status-badge pending">⏳ Pendiente de Subida</span>
            <a href="CASO_3/index.html" class="case-btn sec-btn">Acceder al Caso 3 ➡</a>
          </div>
        </div>
      </div>

    </div>

    <div class="team-card">
      <h3>👥 Integrantes del Equipo de Desarrollo (Presentado Por)</h3>
      <div class="team-grid">
        
        <div class="team-member">
          <div class="team-avatar">CJ</div>
          <div class="team-info">
            <div class="name">Coavoy Cruz Joseph Gabriel</div>
            <div class="role">Estudiante de Ingeniería de Sistemas</div>
          </div>
        </div>

        <div class="team-member">
          <div class="team-avatar">CM</div>
          <div class="team-info">
            <div class="name">Cuchuyrrumi Mamani Manuel Rodrigo</div>
            <div class="role">Estudiante de Ingeniería de Sistemas</div>
          </div>
        </div>

        <div class="team-member">
          <div class="team-avatar">HR</div>
          <div class="team-info">
            <div class="name">Huallpatuiro Rafaile Brayan</div>
            <div class="role">Estudiante de Ingeniería de Sistemas</div>
          </div>
        </div>

        <div class="team-member">
          <div class="team-avatar">MA</div>
          <div class="team-info">
            <div class="name">Mamani Acuña Frank Joseph</div>
            <div class="role">Estudiante de Ingeniería de Sistemas</div>
          </div>
        </div>

      </div>
    </div>

  </main>

  <footer>
    <p><strong>Universidad Andina del Cusco</strong> — Escuela Profesional de Ingeniería de Sistemas</p>
    <p class="footer-sub">Docente: Espetia Huamanga, Hugo | Asignatura: Inteligencia Artificial — Cusco, Perú 2026-II</p>
  </footer>

</body>
</html>
"""

# Escribir index.html en la raíz
with open(r"c:\Users\Usuario\Desktop\OPENCODE\index.html", "w", encoding="utf-8") as f:
    f.write(get_root_index_html())

print("Portada index.html de la raíz actualizada exitosamente.")
