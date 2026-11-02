"""
check_data.py
=============
Revisa la calidad del dataset ANTES de entrenar: cuántas muestras hay por
letra, trazos demasiado cortos o sospechosamente largos, y puntos fuera del
lienzo (señal de que `--canvas` no coincide con el lienzo de grabación).

Uso (desde la raíz del proyecto):
    python training/check_data.py --dataset data/glifo-dataset-2026-11-01.json
    python training/check_data.py --meta 40        # exige 40 muestras por letra
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import DEFAULT_CANVAS, find_dataset, load_dataset  # noqa: E402
from src.FeatureExtractor import (  # noqa: E402
    ALPHABET,
    DEFAULT_K,
    DEFAULT_SMOOTH_PASSES,
    DEFAULT_STEP,
    VALID_K,
    FeatureExtractor,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Revisión de calidad del dataset")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--canvas", type=float, default=DEFAULT_CANVAS)
    ap.add_argument("--k", type=int, default=DEFAULT_K, choices=VALID_K)
    ap.add_argument("--step", type=float, default=DEFAULT_STEP)
    ap.add_argument("--meta", type=int, default=30, help="mínimo de muestras por letra")
    ap.add_argument("--min-symbols", type=int, default=4)
    args = ap.parse_args()

    ruta = Path(args.dataset) if args.dataset else find_dataset()
    datos = load_dataset(ruta, args.canvas)
    fx = FeatureExtractor(k=args.k, step=args.step, smooth_passes=DEFAULT_SMOOTH_PASSES)

    print(f"Dataset: {ruta.name} · lienzo={args.canvas:g}px · K={args.k} · paso={args.step}\n")
    print(f"{'letra':<6}{'n':>4}{'trazos':>8}{'símb.med':>10}{'min':>5}{'max':>5}  avisos")

    problemas = []
    fuera = 0
    for l in ALPHABET:
        muestras = datos.get(l, [])
        largos = []
        for trazos in muestras:
            for t in trazos:
                fuera += sum(1 for x, y in t if not (-0.02 <= x <= 1.02 and -0.02 <= y <= 1.02))
            largos.append(len(fx.extract(fx.join_strokes(trazos), "2d")))
        avisos = []
        if len(muestras) < args.meta:
            avisos.append(f"faltan {args.meta - len(muestras)} muestras")
        if largos:
            med = statistics.median(largos)
            cortas = sum(1 for n in largos if n < args.min_symbols)
            raras = sum(1 for n in largos if n >= args.min_symbols and (n > 2.5 * med or n < med / 2.5))
            if cortas:
                avisos.append(f"{cortas} con <{args.min_symbols} símbolos (se descartan)")
            if raras:
                avisos.append(f"{raras} de longitud atípica (revisar etiqueta)")
        troz = statistics.mean(len(m) for m in muestras) if muestras else 0
        if largos:
            print(f"{l:<6}{len(muestras):>4}{troz:>8.2f}{statistics.mean(largos):>10.1f}{min(largos):>5}{max(largos):>5}  {'; '.join(avisos)}")
        else:
            print(f"{l:<6}{0:>4}{'':>8}{'':>10}{'':>5}{'':>5}  SIN DATOS")
        if avisos:
            problemas.append(l)

    total = sum(len(datos.get(l, [])) for l in ALPHABET)
    print(f"\nTotal: {total} muestras · letras con avisos: {len(problemas)}")
    if fuera:
        print(f"⚠ {fuera} puntos caen fuera del lienzo [0, 1]: revisa --canvas (¿cuál es el lado real del lienzo?).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
