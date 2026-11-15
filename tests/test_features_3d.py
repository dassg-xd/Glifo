import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.FeatureExtractor import (  # noqa: E402
    FeatureExtractor,
    a_esfericas,
    angulo_entre,
    de_esfericas,
    mover,
    rumbo,
)

CENTRO = math.pi / 2   # φ del centro de la vista


def arco(theta0, phi0, theta1, phi1, n=40):
    """n+1 puntos entre dos posiciones (θ, φ), interpolando las coordenadas."""
    return [de_esfericas(theta0 + (theta1 - theta0) * i / n, phi0 + (phi1 - phi0) * i / n) for i in range(n + 1)]


class TestCoordenadas(unittest.TestCase):
    def test_ida_y_vuelta(self):
        for theta in (-1.2, 0.0, 0.7):
            for phi in (0.3, CENTRO, 2.4):
                t, p = a_esfericas(de_esfericas(theta, phi))
                self.assertAlmostEqual(t, theta, places=9)
                self.assertAlmostEqual(p, phi, places=9)

    def test_centro_de_la_vista(self):
        """El punto que mira de frente (0, 0, 1) está en θ = 0, φ = π/2; arriba es φ = 0."""
        self.assertEqual(a_esfericas((0, 0, 1)), (0.0, CENTRO))
        self.assertAlmostEqual(a_esfericas((0, 1, 0))[1], 0.0)

    def test_normaliza_y_valida(self):
        self.assertEqual(a_esfericas((0, 0, 5)), (0.0, CENTRO))
        for malo in ((0, 0, 0), (1, 2), (float("nan"), 0, 1), ("x", 0, 1)):
            with self.assertRaises(ValueError):
                a_esfericas(malo)


class TestRumbo(unittest.TestCase):
    def test_cuatro_direcciones(self):
        fx = FeatureExtractor()
        self.assertEqual(set(fx.extract(arco(-0.3, CENTRO, 0.3, CENTRO), "3d")), {0})       # →
        self.assertEqual(set(fx.extract(arco(0, CENTRO + 0.3, 0, CENTRO - 0.3), "3d")), {2})  # ↑
        self.assertEqual(set(fx.extract(arco(0.3, CENTRO, -0.3, CENTRO), "3d")), {4})       # ←
        self.assertEqual(set(fx.extract(arco(0, CENTRO - 0.3, 0, CENTRO + 0.3), "3d")), {6})  # ↓

    def test_sin_phi_corrige_la_altura(self):
        """Lejos del centro, un Δθ grande recorre poco: sin el factor sin φ un movimiento
        diagonal parecería horizontal. Con él, la diagonal sigue siendo diagonal."""
        p, q = de_esfericas(0.0, 0.6), de_esfericas(0.02, 0.6 - 0.02 * math.sin(0.6))
        self.assertAlmostEqual(rumbo(p, q), math.pi / 4, delta=0.01)
        sin_correccion = math.atan2(0.02 * math.sin(0.6), 0.02)        # usando Δθ crudo
        self.assertLess(sin_correccion, math.pi / 4 - 0.15)

    def test_cruza_el_corte_de_theta(self):
        """θ salta de π a −π detrás de la esfera; el rumbo no debe dar media vuelta."""
        a, b = de_esfericas(math.pi - 0.01, CENTRO), de_esfericas(-math.pi + 0.01, CENTRO)
        self.assertAlmostEqual(rumbo(a, b), 0.0, delta=1e-6)

    def test_mover_es_inverso_de_rumbo(self):
        for direccion in (0.0, 1.0, 2.5, -2.0):
            q = mover((0, 0, 1), direccion, 0.05)
            self.assertAlmostEqual(angulo_entre((0, 0, 1), q), 0.05, places=9)
            self.assertAlmostEqual(rumbo((0, 0, 1), q), direccion, delta=1e-3)


class TestRemuestreo(unittest.TestCase):
    def test_paso_angular(self):
        fx = FeatureExtractor(paso_angular=0.05)
        pts = fx.preprocess(arco(-0.5, CENTRO, 0.5, CENTRO, n=7), "3d")
        self.assertEqual(len(pts), 21)                 # 1 rad / 0.05 = 20 pasos
        for a, b in zip(pts, pts[1:]):
            self.assertAlmostEqual(angulo_entre(a, b), 0.05, places=6)

    def test_velocidad_no_importa(self):
        fx = FeatureExtractor()
        lento, rapido = arco(-0.4, 1.3, 0.4, 1.9, n=200), arco(-0.4, 1.3, 0.4, 1.9, n=9)
        self.assertEqual(fx.extract(lento, "3d"), fx.extract(rapido, "3d"))

    def test_misma_letra_en_otro_lugar_da_lo_mismo(self):
        """Una «L» chica dibujada en el centro o desplazada a un lado da los mismos símbolos."""
        fx = FeatureExtractor()

        def ele(t0, p0):
            return [arco(t0, p0, t0, p0 + 0.3) + arco(t0, p0 + 0.3, t0 + 0.2, p0 + 0.3)[1:]]
        self.assertEqual(fx.extract_trazos(ele(0.0, 1.4), "3d"), fx.extract_trazos(ele(0.5, 1.2), "3d"))

    def test_simbolos_a_puntos(self):
        fx = FeatureExtractor()
        pts = fx.symbols_to_points([0, 0, 2, 2], (0, 0, 1), "3d")
        self.assertEqual(len(pts), 5)
        self.assertEqual(fx.extract(pts, "3d")[:1], [0])


class TestTodaLaEsfera(unittest.TestCase):
    def test_lo_de_atras_tambien_cuenta(self):
        """Un trazo en la cara de atrás (z < 0) da los mismos símbolos que el mismo trazo adelante."""
        fx = FeatureExtractor()
        adelante = arco(-0.4, CENTRO, 0.4, CENTRO)
        atras = arco(math.pi - 0.4, CENTRO, math.pi + 0.4, CENTRO)
        self.assertTrue(all(p[2] < 0 for p in atras))
        self.assertEqual(fx.extract_trazos([atras], "3d"), fx.extract_trazos([adelante], "3d"))

    def test_un_trazo_que_pasa_por_detras_es_un_solo_trazo(self):
        fx = FeatureExtractor()
        vuelta = arco(-0.4, CENTRO, math.pi + 0.4, CENTRO, n=200)
        simbolos, saltos = fx.extract_trazos([vuelta], "3d", con_saltos=True)
        self.assertEqual(saltos, [])
        self.assertTrue(all(s == 0 for s in simbolos))

    def test_arriba_es_el_polo_norte(self):
        """Subir por un meridiano es «arriba» (símbolo 2) en cualquier longitud de la esfera."""
        fx = FeatureExtractor()
        for theta in (0.0, 1.5, math.pi, -2.0):
            self.assertTrue(all(s == 2 for s in fx.extract(arco(theta, 1.9, theta, 1.3), "3d")))

    def test_orden_de_trazos_al_cruzar_el_meridiano_pi(self):
        """Dos palitos a los lados de θ = ±π se ordenan de izquierda a derecha, como en θ = 0."""
        fx = FeatureExtractor()
        for base in (0.0, math.pi):
            izq, der = arco(base - 0.1, 1.4, base - 0.1, 1.8), arco(base + 0.1, 1.4, base + 0.1, 1.8)
            orden = fx.ordenar_trazos_3d([der, izq])
            self.assertLess(angulo_entre(orden[0][0], izq[0]), 1e-9)


class TestVariosTrazos3D(unittest.TestCase):
    def test_orden_no_importa(self):
        """Una «i» con el punto primero o después da la misma secuencia."""
        fx = FeatureExtractor()
        palito, punto = arco(0, 1.6, 0, 1.9, n=10), [de_esfericas(0, 1.45), de_esfericas(0.002, 1.452)]
        self.assertEqual(fx.extract_trazos([palito, punto], "3d"), fx.extract_trazos([punto, palito], "3d"))
        self.assertEqual(fx.extract_trazos([palito, punto], "3d")[-1], 2)   # el salto va hacia arriba


if __name__ == "__main__":
    unittest.main()
