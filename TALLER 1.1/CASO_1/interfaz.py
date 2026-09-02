import os
import sys

import joblib
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELO_PATH = os.path.join(BASE_DIR, "modelo_casas.pkl")

DESCRIPCIONES = {
    "longitude": "Longitud (-124.3 a -114.3)",
    "latitude": "Latitud (32.5 a 41.95)",
    "housing_median_age": "Antigüedad promedio (años)",
    "total_rooms": "Habitaciones totales en el bloque",
    "total_bedrooms": "Dormitorios totales en el bloque",
    "population": "Población total del bloque",
    "households": "Hogares / Familias en el bloque",
    "median_income": "Ingreso medio (decenas de miles $)",
    "ocean_proximity": "Ubicación respecto al océano",
}

OPCIONES_OCEANO = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]

PRESETS = {
    "Bahía de San Francisco (Alto Valor)": {
        "longitude": "-122.23",
        "latitude": "37.88",
        "housing_median_age": "41",
        "total_rooms": "880",
        "total_bedrooms": "129",
        "population": "322",
        "households": "126",
        "median_income": "8.3252",
        "ocean_proximity": "NEAR BAY",
    },
    "Costa de Los Ángeles (Residencial)": {
        "longitude": "-118.45",
        "latitude": "34.02",
        "housing_median_age": "32",
        "total_rooms": "2400",
        "total_bedrooms": "450",
        "population": "1150",
        "households": "430",
        "median_income": "5.6500",
        "ocean_proximity": "<1H OCEAN",
    },
    "Zona Interior de California (Económica)": {
        "longitude": "-119.78",
        "latitude": "36.75",
        "housing_median_age": "25",
        "total_rooms": "1800",
        "total_bedrooms": "380",
        "population": "1400",
        "households": "360",
        "median_income": "2.8500",
        "ocean_proximity": "INLAND",
    },
    "Frente al Océano / Costa Directa": {
        "longitude": "-122.01",
        "latitude": "36.97",
        "housing_median_age": "28",
        "total_rooms": "2100",
        "total_bedrooms": "410",
        "population": "980",
        "households": "390",
        "median_income": "6.2000",
        "ocean_proximity": "NEAR OCEAN",
    },
}


def cargar_modelo():
    try:
        return joblib.load(MODELO_PATH)
    except FileNotFoundError:
        messagebox.showerror(
            "Modelo no encontrado",
            f"No se encontró '{MODELO_PATH}'.\n\nEjecuta primero:  py entrenar.py",
        )
        sys.exit(1)


class InterfazApp:
    def __init__(self, raiz):
        self.raiz = raiz
        datos = cargar_modelo()
        self.pipeline = datos["pipeline"]
        self.columnas_numericas = datos["columnas_numericas"]
        self.metricas = datos.get("metricas", {"r2": 0.657, "rmse": 67043, "mae": 46981})
        self.tipo_modelo = datos.get("tipo_modelo", "Regresión Polinómica (Grado 2)")
        self.campos = {}

        self._configurar_estilos()
        self._configurar_ventana()
        self._crear_encabezado()
        self._crear_selector_presets()
        self._crear_formulario()
        self._crear_panel_prediccion()
        self._crear_panel_metricas()

    def _configurar_estilos(self):
        self.bg_color = "#f8fafc"
        self.card_bg = "#ffffff"
        self.primary = "#2563eb"
        self.primary_hover = "#1d4ed8"
        self.text_dark = "#0f172a"
        self.text_muted = "#64748b"
        self.border_color = "#e2e8f0"
        self.success_color = "#059669"

    def _configurar_ventana(self):
        self.raiz.title("Tasador de Viviendas de California | Regresión Polinómica")
        self.raiz.configure(bg=self.bg_color)
        self.raiz.geometry("640x780")
        self.raiz.minsize(580, 720)

    def _crear_encabezado(self):
        encabezado = tk.Frame(self.raiz, bg=self.bg_color)
        encabezado.pack(padx=24, pady=(18, 10), fill="x")

        tk.Label(
            encabezado,
            text="California Housing Predictor",
            bg=self.bg_color,
            fg=self.text_dark,
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w")

        tk.Label(
            encabezado,
            text=f"Modelo Activo: {self.tipo_modelo} | Transformación no lineal de variables",
            bg=self.bg_color,
            fg=self.text_muted,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(2, 0))

    def _crear_selector_presets(self):
        marco = tk.Frame(self.raiz, bg="#eff6ff", highlightthickness=1, highlightbackground="#bfdbfe")
        marco.pack(padx=24, pady=(0, 12), fill="x")

        tk.Label(
            marco,
            text="Cargar plantilla de ejemplo:",
            bg="#eff6ff",
            fg="#1e40af",
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left", padx=(12, 8), pady=8)

        self.combo_preset = ttk.Combobox(
            marco,
            values=list(PRESETS.keys()),
            state="readonly",
            width=36,
            font=("Segoe UI", 9),
        )
        self.combo_preset.set(list(PRESETS.keys())[0])
        self.combo_preset.pack(side="left", padx=4, pady=8)
        self.combo_preset.bind("<<ComboboxSelected>>", self._aplicar_preset)

        btn_cargar = tk.Button(
            marco,
            text="Aplicar",
            command=self._aplicar_preset,
            bg="#3b82f6",
            fg="white",
            font=("Segoe UI", 8, "bold"),
            relief="flat",
            padx=10,
            pady=2,
            cursor="hand2",
        )
        btn_cargar.pack(side="left", padx=8, pady=8)

    def _aplicar_preset(self, event=None):
        seleccion = self.combo_preset.get()
        if seleccion in PRESETS:
            valores = PRESETS[seleccion]
            for col in self.columnas_numericas:
                if col in self.campos and col in valores:
                    self.campos[col].delete(0, tk.END)
                    self.campos[col].insert(0, valores[col])
            if "ocean_proximity" in valores:
                self.combo_oceano.set(valores["ocean_proximity"])

    def _crear_formulario(self):
        marco_tarjeta = tk.Frame(
            self.raiz,
            bg=self.card_bg,
            highlightthickness=1,
            highlightbackground=self.border_color,
        )
        marco_tarjeta.pack(padx=24, pady=0, fill="both", expand=True)

        canvas = tk.Canvas(marco_tarjeta, bg=self.card_bg, highlightthickness=0)
        scrollbar = ttk.Scrollbar(marco_tarjeta, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.card_bg)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=570)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=12, pady=12)
        scrollbar.pack(side="right", fill="y", pady=12)

        preset_inicial = PRESETS[list(PRESETS.keys())[0]]

        for i, columna in enumerate(self.columnas_numericas):
            fila = tk.Frame(scrollable_frame, bg=self.card_bg)
            fila.pack(fill="x", pady=4)

            etiqueta_texto = columna.replace("_", " ").title()
            tk.Label(
                fila,
                text=f"{etiqueta_texto}:",
                bg=self.card_bg,
                fg=self.text_dark,
                font=("Segoe UI", 9, "bold"),
                width=20,
                anchor="w",
            ).pack(side="left")

            tk.Label(
                fila,
                text=DESCRIPCIONES.get(columna, ""),
                bg=self.card_bg,
                fg=self.text_muted,
                font=("Segoe UI", 8),
            ).pack(side="left", expand=True, fill="x")

            entrada = tk.Entry(
                fila,
                width=12,
                font=("Segoe UI", 10),
                justify="right",
                highlightthickness=1,
                highlightbackground=self.border_color,
            )
            entrada.insert(0, preset_inicial.get(columna, "0"))
            entrada.pack(side="right", padx=(8, 0))
            self.campos[columna] = entrada

        # Fila Ocean Proximity
        fila_prox = tk.Frame(scrollable_frame, bg=self.card_bg)
        fila_prox.pack(fill="x", pady=6)

        tk.Label(
            fila_prox,
            text="Proximidad al Mar:",
            bg=self.card_bg,
            fg=self.text_dark,
            font=("Segoe UI", 9, "bold"),
            width=20,
            anchor="w",
        ).pack(side="left")

        tk.Label(
            fila_prox,
            text=DESCRIPCIONES["ocean_proximity"],
            bg=self.card_bg,
            fg=self.text_muted,
            font=("Segoe UI", 8),
        ).pack(side="left", expand=True, fill="x")

        self.combo_oceano = ttk.Combobox(
            fila_prox,
            values=OPCIONES_OCEANO,
            state="readonly",
            width=14,
            font=("Segoe UI", 9),
        )
        self.combo_oceano.set(preset_inicial.get("ocean_proximity", OPCIONES_OCEANO[0]))
        self.combo_oceano.pack(side="right", padx=(8, 0))

    def _crear_panel_prediccion(self):
        panel = tk.Frame(self.raiz, bg=self.bg_color)
        panel.pack(padx=24, pady=(10, 4), fill="x")

        self.boton = tk.Button(
            panel,
            text="Calcular Valor Estimado",
            command=self.calcular_precio,
            bg=self.primary,
            fg="white",
            activebackground=self.primary_hover,
            activeforeground="white",
            font=("Segoe UI", 11, "bold"),
            cursor="hand2",
            relief="flat",
            pady=8,
        )
        self.boton.pack(fill="x")

        self.resultado = tk.Label(
            panel,
            text="Valor Estimado: $ --",
            bg=self.bg_color,
            fg=self.success_color,
            font=("Segoe UI", 16, "bold"),
        )
        self.resultado.pack(pady=(6, 2))

        self.rango = tk.Label(
            panel,
            text="",
            bg=self.bg_color,
            fg=self.text_muted,
            font=("Segoe UI", 9),
        )
        self.rango.pack()

    def _crear_panel_metricas(self):
        r2 = self.metricas.get("r2", 0.657)
        rmse = self.metricas.get("rmse", 67043)
        mae = self.metricas.get("mae", 46981)
        r2_pct = r2 * 100
        error_relativo = (mae / 206855.0) * 100  # Precio medio de referencia

        marco = tk.Frame(self.raiz, bg="#f1f5f9", highlightthickness=1, highlightbackground=self.border_color)
        marco.pack(fill="x", pady=(8, 14), padx=24)

        texto = (
            f"Precisión del Modelo: R² = {r2_pct:.1f}% (Nivel: Bueno / Sólido ✅)\n"
            f"Error Promedio (MAE): ${mae:,.0f} USD (Error Relativo: {error_relativo:.1f}%) | RMSE: ${rmse:,.0f}"
        )
        tk.Label(
            marco,
            text=texto,
            bg="#f1f5f9",
            fg=self.text_dark,
            font=("Segoe UI", 8, "bold"),
            justify="center",
        ).pack(padx=10, pady=6)

    def _leer_numero(self, texto):
        texto_normalizado = texto.replace(",", ".")
        return float(texto_normalizado)

    def calcular_precio(self):
        try:
            fila = {}
            for columna in self.columnas_numericas:
                valor = self.campos[columna].get().strip()
                if valor == "":
                    raise ValueError(f"El campo '{columna}' está vacío.")
                num = self._leer_numero(valor)
                if num < 0:
                    raise ValueError(f"El valor de '{columna}' no puede ser negativo.")
                fila[columna] = num
            fila["ocean_proximity"] = self.combo_oceano.get()

            # Verificaciones de coherencia lógica
            if fila["total_bedrooms"] > fila["total_rooms"]:
                messagebox.showwarning(
                    "Advertencia de Proporción",
                    "El número de dormitorios no debería ser mayor al total de habitaciones del bloque.",
                )

            if fila["population"] < fila["households"]:
                messagebox.showwarning(
                    "Advertencia de Proporción",
                    "La población suele ser mayor a la cantidad de hogares (promedio: ~2.8 personas por hogar).",
                )

            prediccion = float(self.pipeline.predict(pd.DataFrame([fila]))[0])

            mae = self.metricas.get("mae", 46981)
            min_rango = max(0, prediccion - mae)
            max_rango = prediccion + mae

            self.resultado.configure(text=f"Valor Estimado: ${prediccion:,.0f} USD")
            self.rango.configure(
                text=f"Intervalo Típico (± MAE): ${min_rango:,.0f}  a  ${max_rango:,.0f} USD"
            )
        except ValueError as error:
            messagebox.showerror("Datos inválidos", str(error))
        except Exception as error:
            messagebox.showerror("Error", f"No se pudo calcular el precio:\n{error}")


def main():
    raiz = tk.Tk()
    InterfazApp(raiz)
    raiz.mainloop()


if __name__ == "__main__":
    main()