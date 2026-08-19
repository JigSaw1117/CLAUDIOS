"""
Reemplaza CASO_1/index.html directamente con la interfaz completa de casas_predictor.html,
añadiendo la barra de navegación institucional UAC en la parte superior para permitir
la navegación fluida de regreso a la Portada Principal.
"""

import os

predictor_path = r"c:\Users\Usuario\Desktop\OPENCODE\CASO_1\casas_predictor.html"
index_path = r"c:\Users\Usuario\Desktop\OPENCODE\CASO_1\index.html"

with open(predictor_path, "r", encoding="utf-8") as f:
    content = f.read()

# Insertar el navbar institucional UAC justo después de <body>
nav_html = r"""
<!-- NAVBAR UAC INSTITUCIONAL -->
<div style="width:100%; background:#0a2540; border-bottom:1px solid rgba(0,180,216,0.3); padding:10px 24px; display:flex; justify-content:space-between; align-items:center; box-shadow:0 4px 18px rgba(0,0,0,0.4); margin-bottom: 20px;">
  <a href="../index.html" style="display:flex; align-items:center; gap:12px; text-decoration:none;">
    <img src="logo-uac.jpg" alt="UAC" style="height:44px; width:auto; border-radius:6px; background:white; padding:3px;" />
    <div style="display:flex; flex-direction:column; text-align:left; line-height:1.15;">
      <span style="font-size:14px; font-weight:800; color:#ffffff;">Universidad Andina del Cusco</span>
      <span style="font-size:11px; font-weight:700; color:#00b4d8;">Caso 1 — Tasador de Casas</span>
    </div>
  </a>
  <div style="display:flex; gap:12px; align-items:center;">
    <a href="../index.html" style="color:#cbd5e1; text-decoration:none; font-weight:600; font-size:13.5px; padding:6px 14px; border-radius:8px; background:rgba(255,255,255,0.08); transition:all 0.2s;">🏠 Portada Principal</a>
    <a href="../CASO_2/index.html" style="color:#cbd5e1; text-decoration:none; font-weight:600; font-size:13.5px; padding:6px 14px; border-radius:8px; background:rgba(255,255,255,0.08); transition:all 0.2s;">🍷 Caso 2 (Vinos)</a>
  </div>
</div>
"""

# Reemplazar <body> por <body> + nav_html
content = content.replace("<body>\n", "<body>\n" + nav_html)
content = content.replace("<body>", "<body>\n" + nav_html)

with open(index_path, "w", encoding="utf-8") as f:
    f.write(content)

# Eliminar casas_predictor.html sobrante
if os.path.exists(predictor_path):
    os.remove(predictor_path)

print("CASO_1/index.html actualizado directamente con el predictor de viviendas.")
