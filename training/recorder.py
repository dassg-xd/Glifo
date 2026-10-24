"""
recorder.py
===========
Grabador de trazos para armar el dataset (Tkinter, viene con Python).

Uso (desde la raíz del proyecto):
    python training/recorder.py --autor juan
    python training/recorder.py --autor juan --modo 3d --k 8

Cómo se usa
-----------
1. Elige modo (2D plano / 3D esfera) y la letra que vas a dibujar.
2. Dibuja la letra con el ratón (puedes levantar el lápiz: varios trazos).
3. Revisa la vista previa: los puntos naranja son lo que verá el HMM.
4. Enter = guardar · Esc = borrar el dibujo · Ctrl+Z = deshacer el último guardado.

En modo 3D el disco representa la parte visible de la esfera (vista de cámara,
proyección ortográfica). Es el mismo marco de coordenadas que la GUI debe
entregar al FeatureExtractor, así que los datos grabados aquí sirven para ambos.
"""
from __future__ import annotations

import argparse
import getpass
import math
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import (  # noqa: E402
    count_samples,
    delete_last_sample,
    save_sample,
    screen_to_sphere,
    screen_to_unit,
)
from src.FeatureExtractor import ALPHABET, FeatureExtractor  # noqa: E402

SIZE = 520          # lado del lienzo en píxeles
MARGIN = 20         # margen del disco 3D
META_PERSONAL = 10  # muestras sugeridas por letra y por persona
MIN_SIMBOLOS = 3


class Recorder(tk.Tk):
    def __init__(self, author: str, mode: str, k: int):
        super().__init__()
        self.title("Glifo · Grabador de trazos")
        self.resizable(False, False)
        self.fx = FeatureExtractor(k=k)

        self.mode_var = tk.StringVar(value=mode)
        self.author_var = tk.StringVar(value=author)
        self.letter_var = tk.StringVar(value=ALPHABET[0])
        self.status_var = tk.StringVar(value="Dibuja la letra y pulsa Enter para guardar.")

        self._strokes_px: list = []   # trazos en píxeles (para dibujar)
        self._cur: list | None = None

        self._build_ui()
        self._draw_background()
        self._refresh_info()

        self.bind("<Return>", lambda e: self.save())
        self.bind("<Escape>", lambda e: self.clear())
        self.bind("<Control-z>", lambda e: self.undo_saved())

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        raiz = ttk.Frame(self, padding=10)
        raiz.grid()

        self.canvas = tk.Canvas(raiz, width=SIZE, height=SIZE, bg="white",
                                highlightthickness=1, highlightbackground="#999", cursor="pencil")
        self.canvas.grid(row=0, column=0, rowspan=2)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        panel = ttk.Frame(raiz, padding=(14, 0, 0, 0))
        panel.grid(row=0, column=1, sticky="n")

        ttk.Label(panel, text="Modo").grid(row=0, column=0, sticky="w")
        modos = ttk.Frame(panel)
        modos.grid(row=1, column=0, sticky="w")
        for texto, valor in (("2D plano", "2d"), ("3D esfera", "3d")):
            ttk.Radiobutton(modos, text=texto, value=valor, variable=self.mode_var,
                            command=self._on_mode_change).pack(side="left", padx=(0, 10))

        ttk.Label(panel, text="Autor").grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(panel, textvariable=self.author_var, width=18).grid(row=3, column=0, sticky="w")

        ttk.Label(panel, text="Dibuja la letra").grid(row=4, column=0, sticky="w", pady=(10, 0))
        self.big_letter = tk.Label(panel, textvariable=self.letter_var, font=("Helvetica", 64, "bold"))
        self.big_letter.grid(row=5, column=0)

        nav = ttk.Frame(panel)
        nav.grid(row=6, column=0)
        ttk.Button(nav, text="◀", width=3, command=lambda: self._step_letter(-1)).pack(side="left")
        combo = ttk.Combobox(nav, textvariable=self.letter_var, values=ALPHABET,
                             width=4, state="readonly", justify="center")
        combo.pack(side="left", padx=4)
        combo.bind("<<ComboboxSelected>>", lambda e: self._on_letter_change())
        ttk.Button(nav, text="▶", width=3, command=lambda: self._step_letter(1)).pack(side="left")

        self.count_label = ttk.Label(panel, text="", wraplength=250, justify="left")
        self.count_label.grid(row=7, column=0, sticky="w", pady=(12, 0))

        botones = ttk.Frame(panel)
        botones.grid(row=8, column=0, sticky="w", pady=(10, 0))
        ttk.Button(botones, text="Guardar (Enter)", command=self.save).pack(fill="x")
        ttk.Button(botones, text="Borrar dibujo (Esc)", command=self.clear).pack(fill="x", pady=3)
        ttk.Button(botones, text="Deshacer último guardado", command=self.undo_saved).pack(fill="x")

        self.preview_label = ttk.Label(panel, text="", wraplength=250, justify="left", foreground="#444")
        self.preview_label.grid(row=9, column=0, sticky="w", pady=(12, 0))

        self.missing_label = ttk.Label(panel, text="", wraplength=250, justify="left", foreground="#a15a00")
        self.missing_label.grid(row=10, column=0, sticky="w", pady=(10, 0))

        ttk.Label(raiz, textvariable=self.status_var, foreground="#1a5e1a").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

    # ------------------------------------------------------------ lienzo
    def _draw_background(self) -> None:
        c = self.canvas
        c.delete("all")
        if self.mode_var.get() == "3d":
            cx = cy = SIZE / 2
            r = SIZE / 2 - MARGIN
            c.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#eef3fb", outline="#6b7a99", width=2)
            for yy in (-0.5, 0.0, 0.5):  # paralelos (líneas rectas vistos de frente)
                half = math.sqrt(1 - yy * yy) * r
                c.create_line(cx - half, cy - yy * r, cx + half, cy - yy * r, fill="#c3cde0")
            for phi in (30, 60):          # meridianos
                a = r * math.sin(math.radians(phi))
                c.create_oval(cx - a, cy - r, cx + a, cy + r, outline="#c3cde0")
            c.create_line(cx, cy - r, cx, cy + r, fill="#c3cde0")
        else:
            c.create_rectangle(1, 1, SIZE - 1, SIZE - 1, outline="#bbb")

    def _on_mode_change(self) -> None:
        self.clear()
        self._draw_background()
        self._refresh_info()

    def _on_letter_change(self) -> None:
        self.clear()
        self._refresh_info()

    def _step_letter(self, delta: int) -> None:
        i = (ALPHABET.index(self.letter_var.get()) + delta) % len(ALPHABET)
        self.letter_var.set(ALPHABET[i])
        self._on_letter_change()

    # ---------------------------------------------------------- dibujo
    def _on_press(self, e) -> None:
        self._cur = [(e.x, e.y)]
        self._strokes_px.append(self._cur)
        self.canvas.delete("preview")

    def _on_move(self, e) -> None:
        if self._cur is None:
            return
        x0, y0 = self._cur[-1]
        if (e.x - x0) ** 2 + (e.y - y0) ** 2 < 1:
            return
        self.canvas.create_line(x0, y0, e.x, e.y, fill="#1d4ed8", width=3,
                                capstyle=tk.ROUND, tags="stroke")
        self._cur.append((e.x, e.y))

    def _on_release(self, _e) -> None:
        self._cur = None
        self._update_preview()

    # --------------------------------------------- conversión y vista previa
    def _strokes_normalized(self) -> list:
        """Píxeles -> coordenadas del contrato (2D: [0,1]; 3D: esfera unitaria de cámara)."""
        modo = self.mode_var.get()
        salida = []
        for trazo in self._strokes_px:
            if modo == "3d":
                c = SIZE / 2
                salida.append([screen_to_sphere(px, py, c, c, SIZE / 2 - MARGIN) for px, py in trazo])
            else:
                salida.append([screen_to_unit(px, py, SIZE) for px, py in trazo])
        return salida

    def _to_screen(self, p) -> tuple:
        if self.mode_var.get() == "3d":
            c = SIZE / 2
            r = SIZE / 2 - MARGIN
            return (c + p[0] * r, c - p[1] * r)
        return (p[0] * SIZE, p[1] * SIZE)

    def _update_preview(self) -> None:
        self.canvas.delete("preview")
        if not self._strokes_px:
            self.preview_label.config(text="")
            return
        modo = self.mode_var.get()
        puntos = self.fx.join_strokes(self._strokes_normalized())
        try:
            limpios = self.fx.preprocess(puntos, modo)
            simbolos = self.fx.extract(puntos, modo)
        except ValueError:
            return
        for p in limpios:
            x, y = self._to_screen(p)
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#f97316", outline="", tags="preview")
        texto = " ".join(map(str, simbolos[:48])) + (" …" if len(simbolos) > 48 else "")
        self.preview_label.config(text=f"{len(simbolos)} símbolos (K={self.fx.k}):\n{texto}")

    # --------------------------------------------------------- acciones
    def clear(self) -> None:
        self._strokes_px = []
        self._cur = None
        self.canvas.delete("stroke")
        self.canvas.delete("preview")
        self.preview_label.config(text="")

    def save(self) -> None:
        if not self._strokes_px:
            self.status_var.set("No hay nada que guardar: dibuja primero.")
            return
        modo = self.mode_var.get()
        strokes = self._strokes_normalized()
        puntos = self.fx.join_strokes(strokes)
        try:
            n = len(self.fx.extract(puntos, modo))
        except ValueError:
            n = 0
        if n < MIN_SIMBOLOS:
            self.status_var.set(f"Trazo demasiado corto ({n} símbolos): no se guardó. Dibuja más grande.")
            return
        letra = self.letter_var.get()
        ruta = save_sample(letra, modo, strokes, self.author_var.get())
        self.status_var.set(f"Guardado «{letra}» ({modo.upper()}, {n} símbolos) → {ruta.name}")
        self.clear()
        self._refresh_info()

    def undo_saved(self) -> None:
        letra = self.letter_var.get()
        borrado = delete_last_sample(letra, self.mode_var.get(), self.author_var.get())
        if borrado:
            self.status_var.set(f"Se borró la última muestra de «{letra}» ({borrado.name}).")
        else:
            self.status_var.set(f"No hay muestras tuyas de «{letra}» para deshacer.")
        self._refresh_info()

    def _refresh_info(self) -> None:
        modo = self.mode_var.get()
        autor = self.author_var.get()
        letra = self.letter_var.get()
        total = count_samples(modo)
        mias = count_samples(modo, author=autor)
        self.count_label.config(
            text=f"«{letra}» en {modo.upper()}: {mias[letra]} tuyas (meta {META_PERSONAL}) · {total[letra]} del equipo"
        )
        faltan = [f"{l}({mias[l]})" for l in ALPHABET if mias[l] < META_PERSONAL]
        if faltan:
            self.missing_label.config(text=f"Te faltan letras en {modo.upper()}:\n" + " ".join(faltan))
        else:
            self.missing_label.config(text=f"¡Meta personal completa en {modo.upper()}!")


def main() -> None:
    ap = argparse.ArgumentParser(description="Grabador de trazos para el dataset de Glifo")
    ap.add_argument("--autor", default=None, help="tu nombre (se usa en el nombre de los archivos)")
    ap.add_argument("--modo", choices=["2d", "3d"], default="2d")
    ap.add_argument("--k", type=int, default=8, help="sectores angulares de la vista previa")
    args = ap.parse_args()
    autor = args.autor or getpass.getuser()
    Recorder(autor, args.modo, args.k).mainloop()


if __name__ == "__main__":
    main()
