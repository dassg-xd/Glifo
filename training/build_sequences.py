"""
build_sequences.py
==================
Convierte los trazos crudos de data/<modo>/<letra>/*.json en secuencias de
símbolos discretos que el Integrante 1 usa para entrenar los HMM (Baum-Welch).

Uso (desde la raíz del proyecto):
    python training/build_sequences.py --modo 2d --k 8
    python training/build_sequences.py --modo 3d --k 8 --paso 0.08

Salida: data/secuencias_<modo>_k<K>.json
    {
      "modo": "2d", "k": 8, "paso": 0.04, "n_simbolos": 8,
      "secuencias": { "a": [[0, 0, 1, ...], ...], "b": [...], ... }
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import DATA_DIR, load_samples  # noqa: E402
from src.FeatureExtractor import ALPHABET, FeatureExtractor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Trazos crudos -> secuencias de símbolos")
    ap.add_argument("--modo", choices=["2d", "3d"], default="2d")
    ap.add_argument("--k", type=int, default=8, help="sectores angulares (8 o 16)")
    ap.add_argument("--paso", type=float, default=None, help="paso de remuestreo (por defecto el del extractor)")
    ap.add_argument("--min-simbolos", type=int, default=4, help="descarta trazos con menos símbolos")
    ap.add_argument("--salida", default=None, help="ruta del JSON de salida")
    args = ap.parse_args()

    fx = FeatureExtractor(k=args.k)
    if args.paso is not None:
        if args.modo == "3d":
            fx.step_3d = args.paso
        else:
            fx.step_2d = args.paso
    paso = fx.step_3d if args.modo == "3d" else fx.step_2d

    muestras = load_samples(args.modo)
    if not muestras:
        print(f"No hay muestras en {DATA_DIR / args.modo}. Graba trazos con training/recorder.py primero.")
        return 1

    secuencias = {l: [] for l in ALPHABET}
    descartadas = 0
    for m in muestras:
        puntos = fx.join_strokes(m["strokes"])
        simbolos = fx.extract(puntos, args.modo)
        if len(simbolos) < args.min_simbolos:
            descartadas += 1
            continue
        secuencias[m["letter"]].append(simbolos)

    print(f"Modo {args.modo.upper()} · K={args.k} · paso={paso}")
    print(f"{'letra':<6}{'n':>4}{'media':>8}{'min':>6}{'max':>6}")
    sin_datos = []
    for l in ALPHABET:
        seqs = secuencias[l]
        if not seqs:
            sin_datos.append(l)
            continue
        largos = [len(s) for s in seqs]
        print(f"{l:<6}{len(seqs):>4}{sum(largos) / len(largos):>8.1f}{min(largos):>6}{max(largos):>6}")
    print(f"\nTotal: {sum(len(s) for s in secuencias.values())} secuencias · descartadas por cortas: {descartadas}")
    if sin_datos:
        print("Letras SIN datos:", " ".join(sin_datos))

    salida = Path(args.salida) if args.salida else DATA_DIR / f"secuencias_{args.modo}_k{args.k}.json"
    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(
            {"modo": args.modo, "k": args.k, "paso": paso, "n_simbolos": args.k, "secuencias": secuencias},
            f,
            ensure_ascii=False,
        )
    print(f"Guardado en {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
