import math
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from build_sequences_3d import RANGO_PHI, colocacion_al_azar, repartir_muestras  # noqa: E402
from src.FeatureExtractor import ANGULO_LIENZO, FeatureExtractor, angulo_entre, de_esfericas, pegar_en_esfera  # noqa: E402


def ele():
    """Una «L» en el lienzo 2D (palito hacia abajo y pie hacia la derecha)."""
    palito = [(0.4, 0.3 + 0.01 * i) for i in range(41)]
    pie = [(0.4 + 0.01 * i, 0.7) for i in range(1, 21)]
    return [palito + pie]


class TestPegar(unittest.TestCase):
    def test_centro_y_tamano(self):
        """El centro del lienzo cae en (θ0, φ0) y la distancia al centro se conserva."""
        trazos = pegar_en_esfera([[(0.5, 0.5), (1.0, 0.5)]], 0.2, 1.4, 1.0)
        c = de_esfericas(0.2, 1.4)
        self.assertAlmostEqual(angulo_entre(trazos[0][0], c), 0.0, places=9)
        self.assertAlmostEqual(angulo_entre(trazos[0][1], c), 0.5, places=9)

    def test_mismos_simbolos_que_en_2d(self):
        """Pegada en el centro con el tamaño de referencia, la «L» da lo mismo que en el plano."""
        fx = FeatureExtractor()
        en_esfera = pegar_en_esfera(ele(), 0.0, math.pi / 2, ANGULO_LIENZO)
        self.assertEqual(fx.extract_trazos(en_esfera, "3d"), fx.extract_trazos(ele(), "2d"))

    def test_colocaciones_en_toda_la_esfera(self):
        """Las muestras se pegan en cualquier longitud (también en la cara de atrás) y en una
        franja de latitudes alrededor del ecuador."""
        rng = random.Random(1)
        centros = [de_esfericas(*colocacion_al_azar(rng)[:2]) for _ in range(400)]
        self.assertTrue(any(c[2] < -0.5 for c in centros))
        self.assertTrue(any(c[2] > 0.5 for c in centros))
        self.assertTrue(all(abs(c[1]) < math.sin(RANGO_PHI) + 1e-9 for c in centros))


class TestReparto(unittest.TestCase):
    def test_sin_fuga_y_reproducible(self):
        datos = {"a": [[[(i / 100, 0.5)]] for i in range(50)]}
        tr1, te1 = repartir_muestras(datos, 0.2, 42)
        tr2, te2 = repartir_muestras(datos, 0.2, 42)
        self.assertEqual((tr1, te1), (tr2, te2))
        self.assertEqual(len(te1["a"]), 10)
        self.assertFalse({str(m) for m in tr1["a"]} & {str(m) for m in te1["a"]})


if __name__ == "__main__":
    unittest.main()
