import tkinter as tk
from tkinter import ttk


class ConvertidorTemperatura:
    def __init__(self, ventana):
        self.ventana = ventana
        ventana.title("Convertidor de temperatura")
        ventana.configure(bg="#F4F5F7")
        ventana.resizable(False, False)

        self.color_fondo = "#F4F5F7"
        self.color_acento = "#1F3A93"
        self.color_texto = "#1B1F27"
        self.color_mutado = "#5A6472"
        self.color_resultado = "#E8ECF5"

        self.modos = {
            "Celsius a Fahrenheit": {
                "unidad": "Grados Celsius",
                "simbolo": "°C",
                "extra": "°F",
                "formula": "°F = (°C × 9/5) + 32",
                "convertir": lambda x: (x * 9 / 5) + 32,
            },
            "Celsius a Kelvin": {
                "unidad": "Grados Celsius",
                "simbolo": "°C",
                "extra": "K",
                "formula": "K = °C + 273.15",
                "convertir": lambda x: x + 273.15,
            },
            "Fahrenheit a Celsius": {
                "unidad": "Grados Fahrenheit",
                "simbolo": "°F",
                "extra": "°C",
                "formula": "°C = (°F - 32) × 5/9",
                "convertir": lambda x: (x - 32) * 5 / 9,
            },
            "Fahrenheit a Kelvin": {
                "unidad": "Grados Fahrenheit",
                "simbolo": "°F",
                "extra": "K",
                "formula": "K = (°F - 32) × 5/9 + 273.15",
                "convertir": lambda x: (x - 32) * 5 / 9 + 273.15,
            },
            "Kelvin a Celsius": {
                "unidad": "Kelvin",
                "simbolo": "K",
                "extra": "°C",
                "formula": "°C = K - 273.15",
                "convertir": lambda x: x - 273.15,
            },
            "Kelvin a Fahrenheit": {
                "unidad": "Kelvin",
                "simbolo": "K",
                "extra": "°F",
                "formula": "°F = K × 9/5 - 459.67",
                "convertir": lambda x: x * 9 / 5 - 459.67,
            },
        }

        self.ventana.geometry("560x460")
        ventana.minsize(540, 440)

        self.marco = tk.Frame(ventana, bg=self.color_fondo, padx=28, pady=24)
        self.marco.pack(fill="both", expand=True)

        tk.Label(
            self.marco,
            text="Convertidor de temperaturas",
            bg=self.color_fondo,
            fg=self.color_texto,
            font=("Georgia", 18, "bold"),
        ).pack(anchor="w")

        tk.Label(
            self.marco,
            text="Selecciona una conversion, ingresa un valor y presiona Convertir.",
            bg=self.color_fondo,
            fg=self.color_mutado,
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(4, 18))

        self._construir_selector()
        self._construir_campos()
        self._construir_resultado()
        self._construir_historial()

        self._actualizar_campos()
        self.entrada.focus_set()

    def _construir_selector(self):
        ttk.Label(
            self.marco,
            text="Tipo de conversion",
            background=self.color_fondo,
            foreground=self.color_texto,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")

        self.modo = ttk.Combobox(
            self.marco,
            values=list(self.modos.keys()),
            state="readonly",
            font=("Segoe UI", 10),
        )
        self.modo.current(0)
        self.modo.pack(fill="x", pady=(6, 4))
        self.modo.bind("<<ComboboxSelected>>", lambda e: self._actualizar_campos())

    def _construir_campos(self):
        fila = tk.Frame(self.marco, bg=self.color_fondo)
        fila.pack(fill="x", pady=(10, 0))

        self.label_unidad = tk.Label(
            fila,
            bg=self.color_fondo,
            fg=self.color_texto,
            font=("Segoe UI", 11),
        )
        self.label_unidad.pack(side="left")

        self.entrada = tk.Entry(
            fila,
            width=14,
            justify="center",
            font=("Consolas", 13),
            relief="solid",
            bd=1,
        )
        self.entrada.pack(side="left", padx=(12, 8))
        self.entrada.bind("<Return>", self.convertir)

        tk.Button(
            fila,
            text="Convertir",
            command=self.convertir,
            bg=self.color_acento,
            fg="white",
            activebackground="#2C4FB0",
            activeforeground="white",
            relief="flat",
            padx=18,
            pady=6,
            font=("Segoe UI", 11, "bold"),
            cursor="hand2",
        ).pack(side="left")

    def _construir_resultado(self):
        self.resultado = tk.Label(
            self.marco,
            text="",
            bg=self.color_resultado,
            fg=self.color_texto,
            font=("Georgia", 14, "bold"),
            padx=16,
            pady=14,
            anchor="w",
            justify="left",
        )
        self.resultado.pack(fill="x", pady=(16, 2))

        self.label_formula = tk.Label(
            self.marco,
            bg=self.color_fondo,
            fg=self.color_mutado,
            font=("Consolas", 10),
            anchor="w",
        )
        self.label_formula.pack(fill="x", pady=(2, 0))

        botones = tk.Frame(self.marco, bg=self.color_fondo)
        botones.pack(fill="x", pady=(12, 0))

        tk.Button(
            botones,
            text="Limpiar",
            command=self.limpiar,
            bg="#DFE2E8",
            fg=self.color_texto,
            relief="flat",
            padx=14,
            pady=4,
            font=("Segoe UI", 10),
            cursor="hand2",
        ).pack(side="left")

        tk.Button(
            botones,
            text="Limpiar historial",
            command=self.limpiar_historial,
            bg="#DFE2E8",
            fg=self.color_texto,
            relief="flat",
            padx=14,
            pady=4,
            font=("Segoe UI", 10),
            cursor="hand2",
        ).pack(side="left", padx=(8, 0))

        tk.Button(
            botones,
            text="Salir",
            command=self.ventana.destroy,
            bg="#DFE2E8",
            fg=self.color_texto,
            relief="flat",
            padx=14,
            pady=4,
            font=("Segoe UI", 10),
            cursor="hand2",
        ).pack(side="right")

    def _construir_historial(self):
        tk.Label(
            self.marco,
            text="Historial",
            bg=self.color_fondo,
            fg=self.color_texto,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=(16, 2))

        marco_lista = tk.Frame(self.marco, bg=self.color_fondo, relief="solid", bd=1)
        marco_lista.pack(fill="both", expand=True)

        self.historial_lista = tk.Listbox(
            marco_lista,
            font=("Consolas", 10),
            bg="#FFFFFF",
            fg=self.color_texto,
            activestyle="none",
            selectbackground="#D5DDF0",
            height=6,
            justify="left",
        )
        self.historial_lista.pack(side="left", fill="both", expand=True)

        scroll = ttk.Scrollbar(marco_lista, command=self.historial_lista.yview)
        scroll.pack(side="right", fill="y")
        self.historial_lista.config(yscrollcommand=scroll.set)

    def _actualizar_campos(self, evento=None):
        modo = self.modos[self.modo.get()]
        self.label_unidad.config(text=modo["unidad"])
        self.label_formula.config(text="Formula: " + modo["formula"])
        self.entrada.delete(0, tk.END)

    def _formatear(self, valor):
        redondeado = round(valor, 2)
        if redondeado.is_integer():
            return str(int(redondeado))
        return str(redondeado)

    def convertir(self, evento=None):
        texto = self.entrada.get().strip().replace(",", ".")
        if not texto:
            self.resultado.configure(text="Escribe un valor numerico para convertir.")
            return
        try:
            valor = float(texto)
        except ValueError:
            self.resultado.configure(
                text="Entrada invalida: usa numeros, ej. 25, -10.5 o 300"
            )
            return

        simbolo = self.modos[self.modo.get()]["simbolo"]
        extra = self.modos[self.modo.get()]["extra"]
        convertir = self.modos[self.modo.get()]["convertir"]

        resultado = convertir(valor)
        entrada_fmt = self._formatear(valor)
        resultado_fmt = self._formatear(resultado)

        self.resultado.config(
            text=f"{entrada_fmt} {simbolo}  =  {resultado_fmt} {extra}"
        )

        self.historial_lista.insert(0, f"{entrada_fmt} {simbolo} -> {resultado_fmt} {extra}")
        self.historial_lista.see(0)

    def limpiar(self):
        self.entrada.delete(0, tk.END)
        self.resultado.config(text="")

    def limpiar_historial(self):
        self.historial_lista.delete(0, tk.END)


def main():
    ventana = tk.Tk()
    ConvertidorTemperatura(ventana)
    ventana.mainloop()


if __name__ == "__main__":
    main()