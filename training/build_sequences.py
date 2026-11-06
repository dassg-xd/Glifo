"""
build_sequences.py
==================
Convierte el dataset de trazos (JSON del equipo) en secuencias de símbolos
discretos para entrenar los HMM (Baum-Welch) y evaluarlos.

Uso (desde la raíz del proyecto):
    python training/build_sequences.py --dataset data/glifo-dataset-2026-11-01.json
    python training/build_sequences.py --k 16 --step 0.03 --holdout 0

Salida: data/secuencias_2d_k<K>.json
    {
      "modo": "2d", "k": 8, "n_simbolos": 8, "paso": 0.03, "suavizado": 1,
      "canvas": 1000.0, "holdout": 0.2, "semilla": 42,
      "secuencias": { "a": [[0, 0, 1, ...], ...], ... },   # ENTRENAMIENTO
      "prueba":     { "a": [[...], ...], ... }               # EVALUACIÓN (vacío si --holdout 0)
    }

Este archivo es GENERADO: hay que volver a crearlo cada vez que cambie el dataset
(o K, el paso o el suavizado); si no, queda desactualizado sin que nadie lo note.

La partición es por letra y reproducible (misma semilla -> mismo reparto). El
JSON no dice quién dibujó cada muestra, así que el reparto es aleatorio: la
precisión sobre "prueba" puede ser algo optimista porque trazos de la misma
persona se parecen. Con autores identificados conviene separar por autor.
"""
from __future__ import annotations

import argparse
import json
import random
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


def to_sequences(datos: dict, fx: FeatureExtractor, min_symbols: int):
    """{letra: [muestra]} -> ({letra: [[símbolos]]}, n_descartadas)."""
    secuencias = {l: [] for l in ALPHABET}
    descartadas = 0
    for letra, muestras in datos.items():
        for trazos in muestras:
            simbolos = fx.extract_trazos(trazos, "2d")
            if len(simbolos) < min_symbols:
                descartadas += 1
                continue
            secuencias[letra].append(simbolos)
    return secuencias, descartadas


def split_holdout(secuencias: dict, holdout: float, seed: int):
    """Reparte cada letra en (entrenamiento, prueba) de forma reproducible."""
    train, test = {}, {}
    for letra in ALPHABET:
        seqs = list(secuencias.get(letra, []))
        random.Random(f"{seed}-{letra}").shuffle(seqs)
        n_test = round(len(seqs) * holdout) if holdout > 0 else 0
        test[letra] = seqs[:n_test]
        train[letra] = seqs[n_test:]
    return train, test


def main() -> int:
    ap = argparse.ArgumentParser(description="Dataset de trazos -> secuencias de símbolos (2D)")
    ap.add_argument("--dataset", default=None, help="JSON del dataset (por defecto el más reciente en data/)")
    ap.add_argument("--k", type=int, default=DEFAULT_K, choices=VALID_K, help="sectores angulares")
    ap.add_argument("--step", type=float, default=DEFAULT_STEP, help="paso de remuestreo (fracción del lienzo)")
    ap.add_argument("--smooth", type=int, default=DEFAULT_SMOOTH_PASSES, help="pasadas de suavizado")
    ap.add_argument("--canvas", type=float, default=DEFAULT_CANVAS,
                    help="escala de las coordenadas del dataset (Glifo Capture usa 0–1000, relativa al lienzo)")
    ap.add_argument("--holdout", type=float, default=0.2, help="fracción para prueba por letra (0 = todo a entrenamiento)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-symbols", type=int, default=4, help="descarta trazos con menos símbolos")
    ap.add_argument("--out", default=None, help="ruta del JSON de salida")
    args = ap.parse_args()
    if not 0.0 <= args.holdout < 1.0:
        ap.error("--holdout debe estar en [0, 1)")

    ruta = Path(args.dataset) if args.dataset else find_dataset()
    datos = load_dataset(ruta, args.canvas)
    fx = FeatureExtractor(k=args.k, step=args.step, smooth_passes=args.smooth)
    secuencias, descartadas = to_sequences(datos, fx, args.min_symbols)
    train, test = split_holdout(secuencias, args.holdout, args.seed)

    print(f"Dataset: {ruta.name} · K={args.k} · paso={args.step} · suavizado={args.smooth} · escala 0–{args.canvas:g}")
    print(f"{'letra':<6}{'train':>6}{'prueba':>7}{'media':>7}{'min':>5}{'max':>5}")
    sin_datos = []
    for l in ALPHABET:
        todas = train[l] + test[l]
        if not todas:
            sin_datos.append(l)
            continue
        largos = [len(s) for s in todas]
        print(f"{l:<6}{len(train[l]):>6}{len(test[l]):>7}{sum(largos) / len(largos):>7.1f}{min(largos):>5}{max(largos):>5}")
    n_tr = sum(len(v) for v in train.values())
    n_te = sum(len(v) for v in test.values())
    print(f"\nEntrenamiento: {n_tr} · prueba: {n_te} · descartadas por cortas: {descartadas}")
    if sin_datos:
        print("Letras SIN datos:", " ".join(sin_datos))

    salida = Path(args.out) if args.out else ROOT / "data" / f"secuencias_2d_k{args.k}.json"
    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(
            {
                "modo": "2d",
                "k": args.k,
                "n_simbolos": args.k,
                "paso": args.step,
                "suavizado": args.smooth,
                "canvas": args.canvas,
                "holdout": args.holdout,
                "semilla": args.seed,
                "secuencias": train,
                "prueba": test,
            },
            f,
            ensure_ascii=False,
        )
    print(f"Guardado en {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
