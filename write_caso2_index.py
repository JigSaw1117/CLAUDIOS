"""Genera CASO_2/index.html agregando la barra de navegación superior conectada a la portada principal y otros casos."""

with open(r"c:\Users\Usuario\Desktop\OPENCODE\TALLER 1\wine_predictor_v2.html", "r", encoding="utf-8") as f:
    content = f.read()

# Insert navbar at top of body
nav_html = """
<nav style="background: rgba(10,37,64,0.95); border-bottom: 1px solid rgba(0,180,216,0.3); padding: 12px 24px; display: flex; justify-content: space-between; align-items: center; position: relative; z-index: 100; backdrop-filter: blur(10px);">
  <div style="font-weight: 800; font-size: 1rem; color: #e0f7f5; display: flex; align-items: center; gap: 8px;">
    <span>🦅 Universidad Andina del Cusco</span>
    <span style="background: rgba(0,180,216,0.2); color: #00b4d8; padding: 2px 8px; border-radius: 100px; font-size: 0.75rem; font-weight: 700;">CASO 2 — VINOS</span>
  </div>
  <div style="display: flex; gap: 18px; font-size: 0.88rem; font-weight: 600;">
    <a href="../index.html" style="color: #b8d4e8; text-decoration: none; transition: color 0.2s;">🏠 Portada Principal</a>
    <a href="../CASO_1/index.html" style="color: #b8d4e8; text-decoration: none; transition: color 0.2s;">📌 Caso 1</a>
    <a href="./index.html" style="color: #60a5fa; text-decoration: none; font-weight: 700; border-bottom: 2px solid #60a5fa;">🍷 Caso 2 (Wine Quality)</a>
    <a href="../CASO_3/index.html" style="color: #b8d4e8; text-decoration: none; transition: color 0.2s;">📈 Caso 3</a>
  </div>
</nav>
"""

updated_content = content.replace("<body>", "<body>\n" + nav_html)

with open(r"c:\Users\Usuario\Desktop\OPENCODE\CASO_2\index.html", "w", encoding="utf-8") as f:
    f.write(updated_content)

print("CASO_2/index.html creado exitosamente.")
