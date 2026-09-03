"""
Script para recrear la Portada Principal raíz (index.html) en c:\\Users\\Usuario\\Desktop\\OPENCODE\\index.html
Sirve como hub central de la Universidad Andina del Cusco para acceder a todos los Talleres y Módulos.
"""

import os

root_index_path = r"c:\Users\Usuario\Desktop\OPENCODE\index.html"

root_index_html = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Universidad Andina del Cusco — Portada Principal IA</title>
  <meta name="description" content="Portal de Proyectos y Talleres de Inteligencia Artificial. Universidad Andina del Cusco." />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet" />
  <style>
    :root {
      --uac-navy: #0d3c6c;
      --uac-dark: #0a2540;
      --uac-cyan: #00b4d8;
      --uac-blue: #0077b6;
      --uac-light-cyan: #e6f9f6;
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

    .top-strip {
      height: 5px;
      background: linear-gradient(90deg, #00b4d8 0%, #48cae4 25%, #ffd166 50%, #f77f00 75%, #d62828 100%);
    }

    /* NAVBAR UAC INSTITUCIONAL */
    header.uac-nav {
      background: var(--uac-dark);
      color: white;
      position: sticky; top: 0; z-index: 100;
      box-shadow: 0 4px 14px rgba(10,37,64,0.25);
    }
    .nav-container {
      max-width: 1240px; margin: 0 auto;
      padding: 10px 24px;
      display: flex; justify-content: space-between; align-items: center;
    }
    .brand-logo {
      display: flex; align-items: center; gap: 14px;
      text-decoration: none; color: white;
    }
    .brand-logo img {
      height: 52px; width: auto;
      border-radius: 6px;
      background: white;
      padding: 3px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .brand-title {
      font-weight: 800; font-size: 1.05rem; color: #ffffff; line-height: 1.2;
    }
    .brand-sub {
      font-size: 0.78rem; color: #90e0ef; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px;
    }

    nav ul {
      display: flex; list-style: none; gap: 8px; align-items: center;
    }
    nav ul li a {
      color: #cbd5e1; text-decoration: none;
      font-weight: 600; font-size: 13.5px;
      padding: 8px 14px; border-radius: 8px;
      transition: all 0.2s ease;
    }
    nav ul li a:hover {
      background: rgba(255,255,255,0.12); color: #ffffff;
    }
    nav ul li a.active {
      background: var(--uac-cyan); color: var(--uac-dark);
      font-weight: 700;
    }

    /* HERO CON FOTO REAL DE FONDO (FONDO.JPG) */
    .hero {
      background: linear-gradient(135deg, rgba(10, 37, 64, 0.88) 0%, rgba(13, 60, 108, 0.82) 100%), url('fondo.jpg') center/cover no-repeat;
      color: white; padding: 65px 24px 85px;
      text-align: center;
      position: relative;
      border-bottom: 5px solid var(--uac-cyan);
    }
    .hero-container { max-width: 1000px; margin: 0 auto; position: relative; z-index: 2; }
    
    .academic-badge {
      display: inline-block;
      background: rgba(0,180,216,0.25); border: 1px solid rgba(0,180,216,0.5);
      padding: 6px 18px; border-radius: 100px;
      font-size: 13px; font-weight: 800; letter-spacing: 1.2px; text-transform: uppercase;
      color: #38bdf8; margin-bottom: 16px; backdrop-filter: blur(4px);
    }
    
    .hero h1 {
      font-size: clamp(2rem, 4vw, 3.2rem); font-weight: 800; line-height: 1.18;
      margin-bottom: 14px; color: #ffffff; text-shadow: 0 2px 12px rgba(0,0,0,0.6);
    }
    
    .hero p {
      font-size: 1.12rem; color: #f1f5f9; max-width: 840px; margin: 0 auto 26px;
      line-height: 1.7; text-shadow: 0 2px 6px rgba(0,0,0,0.5);
    }

    .container { max-width: 1200px; margin: -35px auto 60px; padding: 0 24px; position: relative; z-index: 3; }

    /* TARJETAS DE TALLERES Y MÓDULOS */
    .grid-cards {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 28px;
    }

    .workshop-card {
      background: var(--card-bg); border-radius: var(--radius);
      padding: 32px; box-shadow: 0 10px 30px rgba(10,37,64,0.08);
      border: 1px solid #e2e8f0; border-top: 5px solid var(--uac-navy);
      display: flex; flex-direction: column; justify-content: space-between;
      transition: all 0.25s ease;
    }
    .workshop-card:hover {
      transform: translateY(-4px); box-shadow: 0 14px 35px rgba(10,37,64,0.15);
    }

    .card-tag {
      display: inline-block; background: #e0f2fe; color: #0369a1;
      font-size: 12px; font-weight: 800; padding: 4px 12px; border-radius: 100px;
      margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px;
    }
    .workshop-card h2 {
      font-size: 1.4rem; color: var(--uac-navy); font-weight: 800; margin-bottom: 12px;
    }
    .workshop-card p {
      color: var(--text-muted); font-size: 0.98rem; margin-bottom: 20px; line-height: 1.6;
    }

    .btn-card {
      display: inline-flex; align-items: center; justify-content: center; gap: 8px;
      background: linear-gradient(90deg, var(--uac-navy), var(--uac-blue));
      color: white; text-decoration: none; padding: 13px 24px; border-radius: 10px;
      font-weight: 700; font-size: 0.98rem; transition: all 0.2s ease;
    }
    .btn-card:hover {
      background: linear-gradient(90deg, #0077b6, #00b4d8); color: white;
    }

    footer {
      background: var(--uac-dark); color: #94a3b8; padding: 36px 24px;
      text-align: center; font-size: 0.92rem;
    }
    footer strong { color: white; }
    footer .footer-sub { font-size: 0.84rem; margin-top: 6px; color: #64748b; }
  </style>
</head>
<body>

  <div class="top-strip"></div>

  <!-- NAVBAR INSTITUCIONAL UAC -->
  <header class="uac-nav">
    <div class="nav-container">
      <a href="index.html" class="brand-logo">
        <img src="logo-uac.jpg" alt="Universidad Andina del Cusco" />
        <div style="display:flex; flex-direction:column;">
          <span class="brand-title">Universidad Andina del Cusco</span>
          <span class="brand-sub">Ingeniería de Sistemas — Inteligencia Artificial</span>
        </div>
      </a>
      <nav>
        <ul>
          <li><a href="index.html" class="active">🏠 Portada Principal</a></li>
          <li><a href="TALLER 1.1/portada.html">🤖 Taller 1.1</a></li>
          <li><a href="TALLER 1.2/index.html">🤖 Taller 1.2</a></li>
        </ul>
      </nav>
    </div>
  </header>

  <!-- HERO ACADÉMICO -->
  <section class="hero">
    <div class="hero-container">
      <div class="academic-badge">UNIVERSIDAD ANDINA DEL CUSCO — PORTAL IA</div>
      <h1>Modelado Predictivo, Clasificación y Aplicativos de Inteligencia Artificial</h1>
      <p>Portal Institucional de entrega de los Talleres de Aprendizaje Automático y Regresión Logística — Escuela Profesional de Ingeniería de Sistemas.</p>
    </div>
  </section>

  <!-- CONTENIDO EN GRID DE TALLERES -->
  <main class="container">
    <div class="grid-cards">

      <!-- TALLER 1.1 -->
      <div class="workshop-card">
        <div>
          <span class="card-tag">Taller 1.1</span>
          <h2>Modelos de Regresión Lineal Múltiple & Polinomial</h2>
          <p>Predicción sectorial multisectorial (Casos de estudio: Tasación de Viviendas California, Calidad del Vino Tinto y Regresión Polinomial).</p>
        </div>
        <a href="TALLER 1.1/portada.html" class="btn-card">🤖 Ingresar al Taller 1.1 ➡</a>
      </div>

      <!-- TALLER 1.2 -->
      <div class="workshop-card" style="border-top-color: var(--uac-cyan);">
        <div>
          <span class="card-tag" style="background:#e0f2fe; color:#0284c7;">Taller 1.2</span>
          <h2>Clasificación Binaria con Regresión Logística</h2>
          <p>Modelo de clasificación de la Potabilidad del Agua (Agua Potable vs No Potable) desplegado en el navegador con interfaz interactiva y evaluación de métricas ROC-AUC.</p>
        </div>
        <a href="TALLER 1.2/index.html" class="btn-card" style="background: linear-gradient(90deg, #0077b6, #00b4d8);">🤖 Ingresar al Taller 1.2 ➡</a>
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

with open(root_index_path, "w", encoding="utf-8") as f:
    f.write(root_index_html)

print("Portada principal index.html recreada en la raíz.")
