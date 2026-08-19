html_code = r"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Taller 1 — Inteligencia Artificial | Universidad Andina del Cusco</title>
  <meta name="description" content="Modelado Predictivo Multisectorial con Regresión Lineal Múltiple, Polinomial y Despliegue de Aplicativos de IA. Universidad Andina del Cusco." />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet" />
  <style>
    :root {
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
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    html { scroll-behavior: smooth; }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: #f8fafc;
      color: var(--text-primary);
      line-height: 1.6;
      font-size: 16px;
    }

    /* NAVBAR MENU INSTITUCIONAL */
    header {
      background: var(--uac-dark);
      color: white;
      position: sticky; top: 0; z-index: 100;
      box-shadow: 0 4px 14px rgba(10,37,64,0.15);
    }
    .nav-container {
      max-width: 1240px; margin: 0 auto;
      padding: 12px 24px;
      display: flex; justify-content: space-between; align-items: center;
    }
    .brand-logo {
      display: flex; align-items: center; gap: 10px;
      text-decoration: none;
    }

    nav ul {
      display: flex; list-style: none; gap: 8px; align-items: center;
    }
    nav ul li a {
      color: #cbd5e1; text-decoration: none;
      font-weight: 600; font-size: 14px;
      padding: 8px 16px; border-radius: 8px;
      transition: all 0.2s ease;
    }
    nav ul li a:hover {
      background: rgba(255,255,255,0.1); color: #ffffff;
    }
    nav ul li a.active {
      background: var(--uac-cyan); color: var(--uac-dark);
      font-weight: 700;
    }

    /* HERO ACADÉMICO */
    .hero {
      background: linear-gradient(135deg, var(--uac-dark) 0%, var(--uac-navy) 100%);
      color: white; padding: 50px 24px 70px;
      text-align: center;
      border-bottom: 4px solid var(--uac-cyan);
    }
    .hero-container { max-width: 1000px; margin: 0 auto; }
    
    .academic-badge {
      display: inline-block;
      background: rgba(0,180,216,0.18); border: 1px solid rgba(0,180,216,0.4);
      padding: 6px 18px; border-radius: 100px;
      font-size: 13px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;
      color: #38bdf8; margin-bottom: 18px;
    }
    
    .hero h1 {
      font-size: clamp(2rem, 4vw, 3rem); font-weight: 800; line-height: 1.2;
      margin-bottom: 14px; color: #ffffff;
    }
    
    .hero p {
      font-size: 1.1rem; color: #cbd5e1; max-width: 820px; margin: 0 auto 28px;
      line-height: 1.7;
    }

    .meta-pills {
      display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;
    }
    .meta-pill {
      background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.18);
      padding: 8px 18px; border-radius: 100px; font-size: 14px; color: #f1f5f9;
      font-weight: 500;
    }
    .meta-pill strong { color: #fbbf24; font-weight: 700; }

    /* MAIN CONTAINER */
    .container { max-width: 1180px; margin: -30px auto 60px; padding: 0 24px; position: relative; z-index: 2; }

    /* OBJETIVO CARD */
    .info-card {
      background: var(--card-bg); border-radius: var(--radius);
      padding: 32px; box-shadow: 0 4px 20px rgba(10,37,64,0.06);
      margin-bottom: 36px; border: 1px solid #e2e8f0; border-top: 4px solid var(--uac-navy);
    }
    .info-card h2 {
      font-size: 1.35rem; color: var(--uac-navy); margin-bottom: 14px; font-weight: 800;
      display: flex; align-items: center; gap: 10px;
    }
    .info-card p { color: var(--text-muted); font-size: 1.02rem; line-height: 1.75; }

    /* SECTION TITLE */
    .section-title { text-align: center; margin-bottom: 28px; }
    .section-title h2 { font-size: 1.8rem; color: var(--uac-navy); font-weight: 800; }
    .section-title p { color: var(--text-muted); font-size: 1rem; }

    /* CARDS DE CASOS DE ESTUDIO */
    .cases-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 24px;
      margin-bottom: 44px;
    }

    .case-card {
      background: var(--card-bg); border-radius: var(--radius);
      overflow: hidden; box-shadow: 0 4px 16px rgba(10,37,64,0.05);
      border: 1px solid #e2e8f0; transition: transform 0.2s, box-shadow 0.2s;
      display: flex; flex-direction: column;
    }
    .case-card:hover {
      transform: translateY(-4px); box-shadow: 0 10px 25px rgba(10,37,64,0.1);
    }

    .case-header { padding: 22px 24px; color: white; }
    .case-card.c1 .case-header { background: linear-gradient(135deg, #0d3c6c, #0077b6); }
    .case-card.c2 .case-header { background: linear-gradient(135deg, #4a1525, #8b1a35); }
    .case-card.c3 .case-header { background: linear-gradient(135deg, #1e3a8a, #3b82f6); }

    .case-tag {
      font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: 1.2px; opacity: 0.9; margin-bottom: 4px;
    }
    .case-title { font-size: 1.35rem; font-weight: 800; }

    .case-body { padding: 24px; flex: 1; display: flex; flex-direction: column; justify-content: space-between; }
    .case-desc { color: var(--text-muted); font-size: 0.96rem; margin-bottom: 18px; line-height: 1.6; }
    
    .case-features {
      list-style: none; margin-bottom: 22px; font-size: 0.9rem; color: #334155;
    }
    .case-features li { padding: 6px 0; border-bottom: 1px solid #f1f5f9; display: flex; justify-content: space-between; }
    .case-features li strong { color: var(--uac-navy); font-weight: 700; }

    .status-badge {
      display: inline-flex; align-items: center; gap: 6px;
      padding: 6px 14px; border-radius: 100px; font-size: 13px; font-weight: 700; margin-bottom: 16px;
    }
    .status-badge.completed { background: #d1fae5; color: #065f46; }
    .status-badge.pending { background: #fef3c7; color: #92400e; }

    .case-btn {
      display: inline-flex; align-items: center; justify-content: center; gap: 8px;
      padding: 13px 20px; border-radius: 10px; font-weight: 700; font-size: 15px;
      text-decoration: none; transition: all 0.2s ease; width: 100%;
    }
    .case-btn.active-btn {
      background: #8b1a35; color: white;
    }
    .case-btn.active-btn:hover { background: #6b1428; }
    .case-btn.sec-btn {
      background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1;
    }
    .case-btn.sec-btn:hover { background: #e2e8f0; color: #0f172a; }

    /* INTEGRANTES CARD */
    .team-card {
      background: var(--uac-light-cyan); border: 1px solid #b2f0e8;
      border-radius: var(--radius); padding: 32px; margin-bottom: 40px;
    }
    .team-card h3 {
      color: var(--uac-navy); font-size: 1.3rem; font-weight: 800; margin-bottom: 20px;
      display: flex; align-items: center; gap: 10px;
    }
    .team-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 16px;
    }
    .team-member {
      background: white; border-radius: 12px; padding: 18px 20px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.03); border-left: 4px solid var(--uac-blue);
      display: flex; align-items: center; gap: 14px;
    }
    .team-avatar {
      width: 44px; height: 44px; border-radius: 50%; background: var(--uac-navy);
      color: white; font-weight: 800; font-size: 15px;
      display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    }
    .team-info .name { font-weight: 700; font-size: 0.95rem; color: var(--uac-dark); }
    .team-info .role { font-size: 0.8rem; color: var(--text-muted); font-weight: 500; }

    /* FOOTER */
    footer {
      background: var(--uac-dark); color: #94a3b8; padding: 36px 24px;
      text-align: center; font-size: 0.92rem;
    }
    footer strong { color: white; }
    footer .footer-sub { font-size: 0.84rem; margin-top: 6px; color: #64748b; }
  </style>
</head>
<body>

  <!-- NAVBAR INSTITUCIONAL UAC -->
  <header>
    <div class="nav-container">
      <a href="index.html" class="brand-logo">
        <svg class="uac-logo-img" viewBox="0 0 240 60" xmlns="http://www.w3.org/2000/svg" style="height:46px; width:auto;">
  <g transform="translate(5, 5)">
    <path d="M 25 0 C 38 0, 50 10, 50 25 C 50 40, 38 50, 25 50 C 12 50, 0 40, 0 25 C 0 10, 12 0, 25 0 Z" fill="#0d3c6c"/>
    <path d="M 25 8 L 30 18 L 42 16 L 33 25 L 38 37 L 25 30 L 12 37 L 17 25 L 8 16 L 20 18 Z" fill="#00b4d8"/>
    <circle cx="25" cy="23" r="4" fill="#ffffff"/>
  </g>
  <text x="65" y="24" fill="#ffffff" font-family="'Inter', sans-serif" font-weight="800" font-size="14" letter-spacing="0.5">UNIVERSIDAD ANDINA</text>
  <text x="65" y="40" fill="#00b4d8" font-family="'Inter', sans-serif" font-weight="700" font-size="12" letter-spacing="1.5">DEL CUSCO</text>
  <text x="65" y="51" fill="#a0c4e2" font-family="'Inter', sans-serif" font-weight="500" font-size="8" letter-spacing="0.8">Acreditada Internacionalmente</text>
</svg>
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

  <!-- HERO ACADÉMICO -->
  <section class="hero" id="inicio">
    <div class="hero-container">
      <div class="academic-badge">UNIVERSIDAD ANDINA DEL CUSCO — TALLER 1</div>
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

    <!-- OBJETIVO -->
    <div class="info-card">
      <h2>🎯 Objetivo de la Actividad</h2>
      <p>
        Esta actividad práctica tiene como objetivo que cada equipo de desarrollo aborde el ciclo de vida completo de un proyecto de Inteligencia Artificial. Los equipos deberán analizar, entrenar, evaluar y comparar modelos predictivos basados en <strong>Regresión Lineal Múltiple</strong> y <strong>Regresión Polinomial</strong> aplicando los <strong>tres (3) casos de estudio descritos</strong>. El resultado final se integrará en un único aplicativo interactivo desplegado en la nube e incluirá un informe técnico consolidado.
      </p>
    </div>

    <!-- MATRIZ DE CASOS DE ESTUDIO -->
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

    <!-- EQUIPO DE TRABAJO -->
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

  <!-- FOOTER INSTITUCIONAL -->
  <footer>
    <p><strong>Universidad Andina del Cusco</strong> — Escuela Profesional de Ingeniería de Sistemas</p>
    <p class="footer-sub">Docente: Espetia Huamanga, Hugo | Asignatura: Inteligencia Artificial — Cusco, Perú 2026-II</p>
  </footer>

</body>
</html>
"""

with open(r"c:\Users\Usuario\Desktop\OPENCODE\index.html", "w", encoding="utf-8") as out:
    out.write(html_code)

print("Portada root actualizada exitosamente.")
