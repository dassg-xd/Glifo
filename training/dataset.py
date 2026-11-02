"""
dataset.py
==========
Lector del dataset de trazos del equipo (un único JSON).

Formato de entrada (el que genera el recolector del equipo):

    {
      "a": [                                  # letra
        [                                     # muestra
          [ [481.7, 511.8, 0], [472.9, 505.9, 12], ... ],   # trazo: [x, y, t_ms]
          [ ... ]                                           # (otro trazo si levantó el lápiz)
        ],
        ...
      ],
      "b": [ ... ], ...
    }

Las coordenadas NO son píxeles: Glifo Capture las guarda en una escala de 0 a
1000 relativa al lienzo de captura, igual en cualquier pantalla (500 es siempre
la mitad del lienzo). Y crece hacia abajo. Aquí se NORMALIZAN dividiendo entre
1000 (`canvas`) para quedar en [0, 1]; la GUI debe entregar al FeatureExtractor
puntos en esa misma escala relativa (x/L, y/L con L = lado de su lienzo).
El tiempo `t` se descarta: el remuestreo es por longitud de arco, no por tiempo.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.FeatureExtractor import ALPHABET  # noqa: E402

DEFAULT_CANVAS = 1000.0   # escala de Glifo Capture: coordenadas 0–1000 relativas al lienzo (no son píxeles)


def find_dataset(data_dir: Path = ROOT / "data") -> Path:
    """Devuelve el JSON del dataset más reciente en data/ (glifo-dataset-*.json)."""
    candidatos = sorted(Path(data_dir).glob("glifo-dataset-*.json"))
    if not candidatos:
        raise FileNotFoundError(
            f"No hay ningún glifo-dataset-*.json en {data_dir}. Pasa la ruta con --dataset."
        )
    return candidatos[-1]


def load_dataset(path, canvas: float = DEFAULT_CANVAS) -> dict:
    """Lee el JSON y devuelve {letra: [muestra, ...]}.

    Cada muestra es una lista de trazos, y cada trazo una lista de puntos (x, y)
    ya normalizados (x/canvas, y/canvas). Lanza ValueError con la ubicación exacta
    si algo está mal formado (letra desconocida, punto con menos de 2 valores,
    NaN/inf...). Los trazos vacíos y las muestras sin puntos se descartan.
    """
    if not math.isfinite(canvas) or canvas <= 0:
        raise ValueError("canvas debe ser un número finito y positivo.")
    with open(path, encoding="utf-8") as f:
        crudo = json.load(f)
    if not isinstance(crudo, dict):
        raise ValueError("El JSON debe ser un objeto {letra: [muestras]}.")

    datos: dict = {}
    for letra, muestras in crudo.items():
        if letra not in ALPHABET:
            raise ValueError(f"Letra desconocida en el dataset: {letra!r}.")
        if not isinstance(muestras, list):
            raise ValueError(f"Letra {letra!r}: se esperaba una lista de muestras.")
        limpias = []
        for i, muestra in enumerate(muestras):
            trazos = []
            for j, trazo in enumerate(muestra):
                pts = []
                for k, p in enumerate(trazo):
                    where = f"letra {letra!r}, muestra {i}, trazo {j}, punto {k}"
                    if not isinstance(p, (list, tuple)) or len(p) < 2:
                        raise ValueError(f"{where}: se esperaba [x, y, t] y llegó {p!r}.")
                    try:
                        x, y = float(p[0]), float(p[1])
                    except (TypeError, ValueError):
                        raise ValueError(f"{where}: coordenadas no numéricas {p!r}.") from None
                    if not (math.isfinite(x) and math.isfinite(y)):
                        raise ValueError(f"{where}: valor no finito {p!r}.")
                    pts.append((x / canvas, y / canvas))
                if pts:
                    trazos.append(pts)
            if trazos:
                limpias.append(trazos)
        datos[letra] = limpias
    return datos


def count_by_letter(datos: dict) -> dict:
    """{'a': 73, ...} con las 27 letras (0 si falta)."""
    return {l: len(datos.get(l, [])) for l in ALPHABET}
