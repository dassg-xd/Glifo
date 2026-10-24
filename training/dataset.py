"""
dataset.py
==========
Utilidades de datos SIN interfaz gráfica: guardar, leer y contar trazos.

Formato en disco (un archivo por muestra, para que git no genere conflictos):

    data/<modo>/<letra>/<autor>_<AAAAMMDD_HHMMSS_micros>.json

    {
      "version": 1,
      "letter": "a",
      "mode": "2d",                      # o "3d"
      "author": "juan",
      "created": "2026-10-24T17:40:00",
      "strokes": [ [[x, y], ...], ... ]  # 2D: x,y en [0,1] (y hacia abajo)
    }                                    # 3D: [x, y, z] sobre la esfera unitaria

Se guardan los PUNTOS CRUDOS (no los símbolos) para poder cambiar K, el paso de
remuestreo o el suavizado sin volver a dibujar nada.
"""
from __future__ import annotations

import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.FeatureExtractor import ALPHABET  # noqa: E402

DATA_DIR = ROOT / "data"
MODES = ("2d", "3d")


def letter_dirname(letter: str) -> str:
    """Nombre de carpeta seguro para cada letra (evita 'ñ' en rutas de Windows/git)."""
    return "enie" if letter == "ñ" else letter


def sample_dir(mode: str, letter: str, data_dir: Path = DATA_DIR) -> Path:
    return Path(data_dir) / mode / letter_dirname(letter)


def _clean_author(author: str) -> str:
    return re.sub(r"[^A-Za-z0-9-]", "", author or "") or "anon"


# ------------------------------------------------------------------ escritura
def save_sample(
    letter: str,
    mode: str,
    strokes: list,
    author: str = "anon",
    data_dir: Path = DATA_DIR,
) -> Path:
    """Guarda una muestra y devuelve la ruta del archivo creado."""
    if letter not in ALPHABET:
        raise ValueError(f"Letra no válida: {letter!r}")
    if mode not in MODES:
        raise ValueError(f"Modo no válido: {mode!r} (use '2d' o '3d')")
    if not strokes or sum(len(s) for s in strokes) < 2:
        raise ValueError("La muestra necesita al menos 2 puntos.")

    autor = _clean_author(author)
    ahora = datetime.now()
    carpeta = sample_dir(mode, letter, data_dir)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"{autor}_{ahora.strftime('%Y%m%d_%H%M%S_%f')}.json"
    payload = {
        "version": 1,
        "letter": letter,
        "mode": mode,
        "author": autor,
        "created": ahora.isoformat(timespec="seconds"),
        "strokes": [[[round(float(c), 5) for c in p] for p in s] for s in strokes],
    }
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    return ruta


def delete_last_sample(
    letter: str, mode: str, author: Optional[str] = None, data_dir: Path = DATA_DIR
) -> Optional[Path]:
    """Borra la muestra más reciente de esa letra (y autor, si se indica)."""
    carpeta = sample_dir(mode, letter, data_dir)
    if not carpeta.exists():
        return None
    patron = f"{_clean_author(author)}_*.json" if author else "*.json"
    archivos = sorted(carpeta.glob(patron), key=lambda p: p.name.split("_", 1)[-1])
    if not archivos:
        return None
    archivos[-1].unlink()
    return archivos[-1]


# ------------------------------------------------------------------- lectura
def load_samples(mode: str, data_dir: Path = DATA_DIR, letter: Optional[str] = None) -> list:
    """Lee todas las muestras de un modo (opcionalmente de una sola letra)."""
    letras = [letter] if letter else ALPHABET
    muestras = []
    for l in letras:
        carpeta = sample_dir(mode, l, data_dir)
        if not carpeta.exists():
            continue
        for ruta in sorted(carpeta.glob("*.json")):
            with open(ruta, encoding="utf-8") as f:
                muestras.append(json.load(f))
    return muestras


def count_samples(mode: str, data_dir: Path = DATA_DIR, author: Optional[str] = None) -> dict:
    """Cuenta muestras por letra: {'a': 12, 'b': 0, ...} (todas las letras aparecen)."""
    conteo = {}
    for l in ALPHABET:
        carpeta = sample_dir(mode, l, data_dir)
        if not carpeta.exists():
            conteo[l] = 0
            continue
        patron = f"{_clean_author(author)}_*.json" if author else "*.json"
        conteo[l] = len(list(carpeta.glob(patron)))
    return conteo


# ------------------------------------------------- mapeo de pantalla (grabador)
def screen_to_unit(px: float, py: float, size: float) -> tuple:
    """Píxel del lienzo 2D -> coordenadas normalizadas (x/L, y/L). Y sigue hacia abajo."""
    return (px / size, py / size)


def screen_to_sphere(px: float, py: float, cx: float, cy: float, radius: float) -> tuple:
    """Píxel dentro del disco de la esfera -> (x, y, z) en el marco de la cámara.

    Proyección ortográfica inversa: x a la derecha, y arriba, z hacia el
    observador. Si el cursor sale del disco se ajusta al borde (z = 0).
    """
    x = (px - cx) / radius
    y = -(py - cy) / radius
    r2 = x * x + y * y
    if r2 > 1.0:
        r = math.sqrt(r2)
        return (x / r, y / r, 0.0)
    return (x, y, math.sqrt(1.0 - r2))
