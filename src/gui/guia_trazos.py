"""
guia_trazos.py
==============
Cómo se traza cada letra, trazo por trazo y en orden. Los caminos salen DEL
DATASET (data/guia_trazos.json, generado con training/generar_guia.py): cada
trazo es el trazo mediano de todas las muestras de la letra, así que la guía
enseña exactamente el diseño único por letra con el que aprenden los modelos.
El mismo archivo, en formato JS, es la guía de dibujo de Glifo Capture.

Coordenadas en el espacio de 0 a 1000 del lienzo (la y crece hacia abajo) y
sobre las mismas líneas guía: ascendente 180, altura x 420, línea base 700,
descendente 900. Cada trazo es un camino SVG con M, L y C; el orden de la lista
es el orden en que se dibuja. Un trazo que vuelve sobre sí mismo (la «n» baja,
sube por el mismo palito y hace el arco) es UN solo trazo.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

RUTA_GUIA = Path(__file__).resolve().parents[2] / "data" / "guia_trazos.json"


def _cargar_guia(ruta: Path = RUTA_GUIA) -> dict[str, list[str]]:
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)["guia"]


GUIA: dict[str, list[str]] = _cargar_guia()

_TOKEN = re.compile(r"[MLC]|-?\d+(?:\.\d+)?")


def segmentos_de_camino(camino: str) -> list:
    """Camino SVG (M, L, C en coordenadas absolutas) -> lista de piezas:
    ("L", p0, p1) o ("C", p0, c1, c2, p1)."""
    tokens = _TOKEN.findall(camino)
    piezas, actual, i, cmd = [], None, 0, None
    while i < len(tokens):
        if tokens[i] in "MLC":
            cmd = tokens[i]
            i += 1
        n = {"M": 2, "L": 2, "C": 6}[cmd]
        nums = [float(t) for t in tokens[i:i + n]]
        i += n
        if cmd == "M":
            actual = (nums[0], nums[1])
        elif cmd == "L":
            fin = (nums[0], nums[1])
            piezas.append(("L", actual, fin))
            actual = fin
        else:
            c1, c2, fin = (nums[0], nums[1]), (nums[2], nums[3]), (nums[4], nums[5])
            piezas.append(("C", actual, c1, c2, fin))
            actual = fin
    return piezas


def _evaluar(pieza, t: float) -> tuple:
    if pieza[0] == "L":
        (x0, y0), (x1, y1) = pieza[1], pieza[2]
        return (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
    _, p0, c1, c2, p1 = pieza
    u = 1 - t
    return tuple(u ** 3 * p0[k] + 3 * u * u * t * c1[k] + 3 * u * t * t * c2[k] + t ** 3 * p1[k] for k in range(2))


def puntos_de_camino(camino: str, paso: float = 4.0) -> list:
    """Puntos a lo largo del camino, más o menos cada `paso` unidades (de 0 a 1000)."""
    pts = []
    for pieza in segmentos_de_camino(camino):
        muestra = [_evaluar(pieza, k / 40) for k in range(41)]
        largo = sum(math.dist(a, b) for a, b in zip(muestra, muestra[1:]))
        n = max(1, int(largo / paso))
        tramo = [_evaluar(pieza, k / n) for k in range(n + 1)]
        pts.extend(tramo if not pts else tramo[1:])
    return pts


def trazos_de_letra(letra: str, paso: float = 4.0) -> list:
    """Los trazos de la letra como listas de puntos normalizados (0..1), en orden."""
    return [[(x / 1000.0, y / 1000.0) for x, y in puntos_de_camino(c, paso)] for c in GUIA[letra]]
