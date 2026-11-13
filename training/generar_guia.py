"""
generar_guia.py
===============
Saca la guía de dibujo DEL PROPIO DATASET: para cada letra calcula un
prototipo (el trazo mediano de todas las muestras) y lo escribe como caminos
SVG suaves, en el formato de la guía de Glifo Capture.

Así la guía enseña exactamente el diseño con el que aprenden los modelos. La
usan la «Guía de letras» de la app, las plantillas de la proyección y, copiada,
la guía de dibujo de Glifo Capture.

Cómo se calcula el prototipo
----------------------------
1. Se toman las muestras con el número de trazos más común de la letra, en
   orden canónico (FeatureExtractor.ordenar_trazos), y se descartan las de
   tamaño atípico (alto fuera de 0.75–1.33 veces la mediana).
2. Cada trazo se remuestrea por longitud de arco al mismo número de puntos y se
   toma la MEDIANA punto a punto (robusta a muestras raras).
3. El trazo se simplifica (Ramer–Douglas–Peucker), se parte en las esquinas
   agudas (como donde la «n» vuelve a subir por el palito) y cada pieza se
   convierte en curvas de Bézier (Catmull-Rom).

Uso (desde la raíz del proyecto):
    python training/generar_guia.py                 # escribe data/guia_trazos.json
    python training/generar_guia.py --js guia.js    # además, el archivo para Glifo Capture

Volver a correrlo cada vez que cambie el dataset.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import DEFAULT_CANVAS, find_dataset, load_dataset  # noqa: E402
from src.FeatureExtractor import ALPHABET, FeatureExtractor  # noqa: E402

PUNTOS_POR_TRAZO = 120       # puntos para un trazo de 0.6 del lienzo (proporcional al largo)
TOLERANCIA_RDP = 0.004       # simplificación (fracción del lienzo)
ANGULO_ESQUINA = 100         # grados: giro a partir del cual hay una esquina
SALIDA = ROOT / "data" / "guia_trazos.json"


def remuestrear(t, n: int) -> np.ndarray:
    t = np.asarray(t, float)
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(t, axis=0), axis=1))] if len(t) > 1 else np.zeros(1)
    if seg[-1] < 1e-9:
        return np.repeat(t[:1], n, axis=0)
    u = np.linspace(0, seg[-1], n)
    return np.c_[np.interp(u, seg, t[:, 0]), np.interp(u, seg, t[:, 1])]


def prototipo(muestras) -> tuple[list, int]:
    """Muestras de una letra (listas de trazos 0..1) -> (trazos del prototipo, muestras usadas)."""
    fx = FeatureExtractor()
    ms = [fx.ordenar_trazos([t for t in m if t]) for m in muestras]
    nt = collections.Counter(len(m) for m in ms).most_common(1)[0][0]
    ms = [m for m in ms if len(m) == nt]
    altos = []
    for m in ms:
        a = np.concatenate([np.asarray(t) for t in m])
        altos.append(a[:, 1].max() - a[:, 1].min())
    med = float(np.median(altos))
    ms = [m for m, h in zip(ms, altos) if 0.75 * med <= h <= 1.33 * med] or ms
    trazos = []
    for k in range(nt):
        largos = [sum(math.dist(a, b) for a, b in zip(m[k], m[k][1:])) for m in ms]
        largo = float(np.median(largos))
        if largo < 0.01:                                   # un punto (el de la «i»)
            trazos.append(np.median(np.stack([np.mean(np.asarray(m[k]), axis=0) for m in ms]), axis=0)[None, :])
            continue
        n = max(8, int(PUNTOS_POR_TRAZO * largo / 0.6))
        trazos.append(np.median(np.stack([remuestrear(m[k], n) for m in ms]), axis=0))
    return trazos, len(ms)


def _rdp(p: np.ndarray, eps: float) -> list:
    """Índices que conserva Ramer–Douglas–Peucker."""
    if len(p) < 3:
        return list(range(len(p)))
    a, b = p[0], p[-1]
    ab = b - a
    n = np.linalg.norm(ab)
    if n < 1e-12:
        d = np.linalg.norm(p - a, axis=1)
    else:
        d = np.abs(ab[0] * (p[:, 1] - a[1]) - ab[1] * (p[:, 0] - a[0])) / n
    i = int(np.argmax(d))
    if d[i] <= eps:
        return [0, len(p) - 1]
    izq = _rdp(p[: i + 1], eps)
    der = _rdp(p[i:], eps)
    return izq[:-1] + [j + i for j in der]


def _a_bezier(p: np.ndarray) -> list:
    """Puntos -> curvas Catmull-Rom como Bézier cúbicas [(c1, c2, fin), ...]."""
    curvas = []
    for i in range(len(p) - 1):
        p0 = p[i - 1] if i > 0 else p[i]
        p1, p2 = p[i], p[i + 1]
        p3 = p[i + 2] if i + 2 < len(p) else p[i + 1]
        curvas.append((p1 + (p2 - p0) / 6, p2 - (p3 - p1) / 6, p2))
    return curvas


def a_camino_svg(trazo: np.ndarray) -> str:
    """Un trazo del prototipo (0..1) -> camino SVG en el espacio 0..1000."""
    def f(q):
        return f"{round(q[0] * 1000)} {round(q[1] * 1000)}"
    if len(trazo) == 1:
        x, y = trazo[0]
        return f"M{f((x, y))} L{f((x, y + 0.001))}"
    clave = trazo[_rdp(trazo, TOLERANCIA_RDP)]
    # partir en las esquinas agudas para que las curvas no se pasen de largo
    cortes = [0]
    for i in range(1, len(clave) - 1):
        u, v = clave[i] - clave[i - 1], clave[i + 1] - clave[i]
        cos = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
        if math.degrees(math.acos(max(-1.0, min(1.0, cos)))) > ANGULO_ESQUINA:
            cortes.append(i)
    cortes.append(len(clave) - 1)
    partes = [f"M{f(clave[0])}"]
    for a, b in zip(cortes, cortes[1:]):
        pieza = clave[a: b + 1]
        if len(pieza) == 2:
            partes.append(f"L{f(pieza[1])}")
        else:
            partes += [f"C{f(c1)} {f(c2)} {f(fin)}" for c1, c2, fin in _a_bezier(pieza)]
    return " ".join(partes)


def generar(ruta_dataset: Path | None = None, canvas: float = DEFAULT_CANVAS) -> dict:
    ruta = ruta_dataset or find_dataset()
    datos = load_dataset(ruta, canvas)
    guia, usadas = {}, {}
    for letra in ALPHABET:
        trazos, n = prototipo(datos[letra])
        guia[letra] = [a_camino_svg(t) for t in trazos]
        usadas[letra] = n
    return {"dataset": ruta.name, "muestras_usadas": usadas, "guia": guia}


def a_js(guia: dict, dataset: str) -> str:
    lineas = [
        "// Guía de dibujo: cómo se traza cada letra, trazo por trazo y en orden.",
        "//",
        f"// GENERADO desde el dataset de Glifo ({dataset}) con training/generar_guia.py del",
        "// repo Glifo: cada trazo es el trazo mediano de todas las muestras de la letra, así que",
        "// la guía enseña exactamente el diseño con el que aprenden los modelos. No editar a mano:",
        "// si cambia el dataset, se vuelve a generar allá y se copia aquí.",
        "//",
        "// Coordenadas en el espacio 0..1000 del lienzo (la y crece hacia abajo), sobre las líneas",
        "// guía: ascendente 180, altura x 420, línea base 700, descendente 900. El orden de la",
        "// lista es el orden en que se dibuja.",
        "",
        "export const GUIA = {",
    ]
    for letra, caminos in guia.items():
        clave = f'"{letra}"' if letra == "ñ" else letra
        lineas.append(f"  {clave}: [" + ", ".join(f'"{c}"' for c in caminos) + "],")
    lineas.append("};")
    return "\n".join(lineas) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Guía de dibujo a partir del dataset (trazo mediano por letra)")
    ap.add_argument("--dataset", default=None, help="JSON del dataset (por defecto el más reciente en data/)")
    ap.add_argument("--out", default=str(SALIDA), help="JSON de salida para la app")
    ap.add_argument("--js", default=None, help="además, escribe el guia.js de Glifo Capture en esta ruta")
    args = ap.parse_args()
    res = generar(Path(args.dataset) if args.dataset else None)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"Guía generada desde {res['dataset']} -> {args.out}")
    print("Muestras usadas por letra:", " ".join(f"{l}:{n}" for l, n in res["muestras_usadas"].items()))
    if args.js:
        Path(args.js).write_text(a_js(res["guia"], res["dataset"]), encoding="utf-8")
        print(f"guia.js para Glifo Capture -> {args.js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
