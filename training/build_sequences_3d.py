"""
build_sequences_3d.py
=====================
Dataset 3D SINTÉTICO a partir del dataset 2D: cada muestra se «pega» sobre la
esfera y de ahí salen las secuencias de símbolos esféricos para entrenar los HMM
del modo 3D.

Cómo se pega una muestra
------------------------
El lienzo 2D (cuadrado de lado 1) se pone sobre el plano tangente de un punto c
de la cara visible de la esfera y se enrolla sobre la superficie con el mapa
exponencial: un punto a distancia r del centro del lienzo, en dirección d, cae a
un ángulo r de c sobre el círculo máximo con esa dirección. Las distancias y
los ángulos medidos desde el centro se conservan; lejos del centro, la
curvatura deforma la letra un poco, que es justo lo que los modelos 3D deben
aprender.

Para que los modelos no dependan de un solo caso, cada muestra se pega varias
veces con valores al azar:
  * centro c: θ ∈ ±0.35 rad y φ ∈ π/2 ± 0.30 rad (siempre en la cara visible),
  * tamaño: el lado del lienzo cubre ANGULO_LIENZO × U(0.7, 1.3) radianes,
  * giro: ±8°, porque «arriba» es el borde superior del panel y la gente no
    dibuja perfectamente derecho.
Lo que quede detrás de la esfera (z ≤ 0) se descarta, como en la GUI.

Uso (desde la raíz del proyecto):
    python training/build_sequences_3d.py
    python training/build_sequences_3d.py --copias 4 --copias-prueba 2

Salida: data/secuencias_3d_k<K>.json (mismo formato que el de 2D, con "modo": "3d").

El reparto entrenamiento/prueba es por MUESTRA y se hace antes de pegarlas:
ninguna muestra de prueba aparece en el entrenamiento, ni siquiera en otra
posición de la esfera.
"""
from __future__ import annotations

import argparse
import json
import math
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
    ANGULO_LIENZO,
    DEFAULT_K,
    DEFAULT_PASO_ANGULAR,
    DEFAULT_SMOOTH_PASSES,
    VALID_K,
    FeatureExtractor,
    base_local,
    de_esfericas,
)

CENTRO_LIENZO = (0.5, 0.5)
RANGO_THETA = 0.35
RANGO_PHI = 0.30
RANGO_ESCALA = (0.7, 1.3)
RANGO_GIRO = math.radians(8)


def pegar_en_esfera(trazos, theta0: float, phi0: float, lado: float, giro: float = 0.0) -> list:
    """Trazos 2D normalizados (0..1, y hacia abajo) -> trazos (x, y, z) sobre la esfera.

    El centro del lienzo cae en (θ0, φ0) y su lado mide `lado` radianes.
    """
    c = de_esfericas(theta0, phi0)
    e, n = base_local(c)
    cg, sg = math.cos(giro), math.sin(giro)
    salida = []
    for trazo in trazos:
        nuevo = []
        for x, y in trazo:
            u = (x - CENTRO_LIENZO[0]) * lado
            w = -(y - CENTRO_LIENZO[1]) * lado            # en pantalla la y crece hacia abajo
            u, w = cg * u - sg * w, sg * u + cg * w
            r = math.hypot(u, w)
            if r < 1e-12:
                nuevo.append(c)
                continue
            d = tuple((u * e[i] + w * n[i]) / r for i in range(3))
            nuevo.append(tuple(math.cos(r) * c[i] + math.sin(r) * d[i] for i in range(3)))
        salida.append(nuevo)
    return salida


def colocacion_al_azar(rng: random.Random) -> tuple:
    """(θ0, φ0, lado, giro) al azar dentro de los rangos del módulo."""
    return (
        rng.uniform(-RANGO_THETA, RANGO_THETA),
        math.pi / 2 + rng.uniform(-RANGO_PHI, RANGO_PHI),
        ANGULO_LIENZO * rng.uniform(*RANGO_ESCALA),
        rng.uniform(-RANGO_GIRO, RANGO_GIRO),
    )


def repartir_muestras(datos: dict, holdout: float, seed: int):
    """{letra: [muestra]} -> (entrenamiento, prueba), por letra y reproducible."""
    train, test = {}, {}
    for letra in ALPHABET:
        muestras = list(datos.get(letra, []))
        random.Random(f"{seed}-{letra}").shuffle(muestras)
        n_test = round(len(muestras) * holdout) if holdout > 0 else 0
        test[letra], train[letra] = muestras[:n_test], muestras[n_test:]
    return train, test


def a_secuencias(muestras_por_letra: dict, fx: FeatureExtractor, copias: int, rng: random.Random,
                 min_symbols: int):
    """Pega cada muestra `copias` veces y extrae sus símbolos esféricos."""
    secuencias = {l: [] for l in ALPHABET}
    descartadas = 0
    for letra in ALPHABET:
        for trazos in muestras_por_letra.get(letra, []):
            for _ in range(copias):
                en_esfera = pegar_en_esfera(trazos, *colocacion_al_azar(rng))
                simbolos = fx.extract_trazos(en_esfera, "3d")
                if len(simbolos) < min_symbols:
                    descartadas += 1
                    continue
                secuencias[letra].append(simbolos)
    return secuencias, descartadas


def main() -> int:
    ap = argparse.ArgumentParser(description="Dataset 2D -> secuencias 3D sintéticas sobre la esfera")
    ap.add_argument("--dataset", default=None, help="JSON del dataset (por defecto el más reciente en data/)")
    ap.add_argument("--k", type=int, default=DEFAULT_K, choices=VALID_K, help="sectores angulares")
    ap.add_argument("--paso", type=float, default=DEFAULT_PASO_ANGULAR, help="paso de remuestreo (radianes)")
    ap.add_argument("--smooth", type=int, default=DEFAULT_SMOOTH_PASSES, help="pasadas de suavizado")
    ap.add_argument("--canvas", type=float, default=DEFAULT_CANVAS, help="escala de las coordenadas del dataset")
    ap.add_argument("--holdout", type=float, default=0.2, help="fracción de MUESTRAS para prueba por letra")
    ap.add_argument("--copias", type=int, default=4, help="veces que se pega cada muestra de entrenamiento")
    ap.add_argument("--copias-prueba", type=int, default=2, help="veces que se pega cada muestra de prueba")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-symbols", type=int, default=4, help="descarta secuencias con menos símbolos")
    ap.add_argument("--out", default=None, help="ruta del JSON de salida")
    args = ap.parse_args()
    if not 0.0 <= args.holdout < 1.0:
        ap.error("--holdout debe estar en [0, 1)")
    if args.copias < 1 or args.copias_prueba < 1:
        ap.error("--copias y --copias-prueba deben ser al menos 1")

    ruta = Path(args.dataset) if args.dataset else find_dataset()
    datos = load_dataset(ruta, args.canvas)
    fx = FeatureExtractor(k=args.k, smooth_passes=args.smooth, paso_angular=args.paso)
    m_train, m_test = repartir_muestras(datos, args.holdout, args.seed)
    rng = random.Random(f"{args.seed}-colocacion")
    train, d1 = a_secuencias(m_train, fx, args.copias, rng, args.min_symbols)
    test, d2 = a_secuencias(m_test, fx, args.copias_prueba, rng, args.min_symbols)

    print(f"Dataset: {ruta.name} · K={args.k} · paso={args.paso:.4f} rad · suavizado={args.smooth}")
    print(f"Copias por muestra: entrenamiento {args.copias}, prueba {args.copias_prueba}")
    print(f"{'letra':<6}{'train':>6}{'prueba':>7}{'media':>7}{'min':>5}{'max':>5}")
    for l in ALPHABET:
        todas = train[l] + test[l]
        if not todas:
            continue
        largos = [len(s) for s in todas]
        print(f"{l:<6}{len(train[l]):>6}{len(test[l]):>7}{sum(largos) / len(largos):>7.1f}{min(largos):>5}{max(largos):>5}")
    n_tr = sum(len(v) for v in train.values())
    n_te = sum(len(v) for v in test.values())
    print(f"\nEntrenamiento: {n_tr} · prueba: {n_te} · descartadas por cortas: {d1 + d2}")

    salida = Path(args.out) if args.out else ROOT / "data" / f"secuencias_3d_k{args.k}.json"
    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(
            {
                "modo": "3d",
                "k": args.k,
                "n_simbolos": args.k,
                "paso": args.paso,
                "angulo_lienzo": ANGULO_LIENZO,
                "suavizado": args.smooth,
                "canvas": args.canvas,
                "holdout": args.holdout,
                "copias": args.copias,
                "copias_prueba": args.copias_prueba,
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
