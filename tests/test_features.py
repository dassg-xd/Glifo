"""
Pruebas de FeatureExtractor (2D).

Ejecutar desde la raíz del proyecto:
    python -m unittest discover -s tests -v
"""
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.FeatureExtractor import (  # noqa: E402
    ALPHABET,
    FeatureExtractor,
    quantize_angle,
    symbol_to_angle,
)


def linea(p0, p1, n):
    """n+1 puntos equiespaciados entre p0 y p1."""
    return [(p0[0] + (p1[0] - p0[0]) * i / n, p0[1] + (p1[1] - p0[1]) * i / n) for i in range(n + 1)]


class TestCuantizacion(unittest.TestCase):
    def test_centros_de_sector(self):
        for k in (8, 16):
            for s in range(k):
                self.assertEqual(quantize_angle(symbol_to_angle(s, k), k), s)

    def test_fronteras_k8(self):
        sector = 2 * math.pi / 8
        self.assertEqual(quantize_angle(sector / 2 - 1e-6, 8), 0)   # justo antes de 22.5°
        self.assertEqual(quantize_angle(sector / 2 + 1e-6, 8), 1)   # justo después

    def test_angulos_negativos_y_mayores_a_2pi(self):
        self.assertEqual(quantize_angle(-math.pi / 2, 8), 6)        # abajo
        self.assertEqual(quantize_angle(2 * math.pi + 0.01, 8), 0)
        self.assertEqual(quantize_angle(-0.01, 8), 0)

    def test_alfabeto_tiene_27_letras(self):
        self.assertEqual(len(ALPHABET), 27)
        self.assertIn("ñ", ALPHABET)
        self.assertEqual(len(set(ALPHABET)), 27)


class TestExtract(unittest.TestCase):
    def setUp(self):
        self.fx = FeatureExtractor(k=8, step=0.04, smooth_passes=0)

    def test_direcciones_cardinales(self):
        # En pantalla Y crece hacia abajo: "arriba" = y decreciente.
        casos = {
            0: ((0.2, 0.5), (0.8, 0.5)),   # →
            2: ((0.5, 0.8), (0.5, 0.2)),   # ↑
            4: ((0.8, 0.5), (0.2, 0.5)),   # ←
            6: ((0.5, 0.2), (0.5, 0.8)),   # ↓
            1: ((0.2, 0.8), (0.8, 0.2)),   # ↗
            3: ((0.8, 0.8), (0.2, 0.2)),   # ↖
            5: ((0.8, 0.2), (0.2, 0.8)),   # ↙
            7: ((0.2, 0.2), (0.8, 0.8)),   # ↘
        }
        for simbolo, (a, b) in casos.items():
            sec = self.fx.extract(linea(a, b, 50))
            self.assertGreater(len(sec), 5)
            self.assertEqual(set(sec), {simbolo}, f"dirección {simbolo}")

    def test_invariante_a_la_velocidad(self):
        lento = linea((0.1, 0.5), (0.9, 0.5), 400)    # muchos eventos del ratón
        rapido = linea((0.1, 0.5), (0.9, 0.5), 7)     # pocos eventos
        self.assertEqual(self.fx.extract(lento), self.fx.extract(rapido))

    def test_longitud_proporcional_al_recorrido(self):
        sec = self.fx.extract(linea((0.1, 0.5), (0.9, 0.5), 100))
        self.assertAlmostEqual(len(sec), 0.8 / 0.04, delta=1)

    def test_puntos_repetidos_no_generan_simbolos(self):
        pts = [(0.5, 0.5)] * 20 + linea((0.5, 0.5), (0.9, 0.5), 30)
        self.assertEqual(set(self.fx.extract(pts)), {0})

    def test_entradas_degeneradas(self):
        self.assertEqual(self.fx.extract([]), [])
        self.assertEqual(self.fx.extract([(0.5, 0.5)]), [])
        self.assertEqual(self.fx.extract([(0.5, 0.5), (0.5, 0.5)]), [])
        self.assertEqual(self.fx.extract([(0.5, 0.5), (0.51, 0.5)]), [])   # menor a un paso

    def test_circulo_antihorario_recorre_los_sectores_en_orden(self):
        pts = [(0.5 + 0.25 * math.cos(2 * math.pi * i / 200),
                0.5 - 0.25 * math.sin(2 * math.pi * i / 200)) for i in range(201)]
        sec = self.fx.extract(pts)
        saltos = {(b - a) % 8 for a, b in zip(sec, sec[1:])}
        self.assertTrue(saltos <= {0, 1}, f"saltos inesperados: {saltos}")
        self.assertGreaterEqual(len(set(sec)), 7)

    def test_suavizado_no_cambia_una_recta(self):
        fx = FeatureExtractor(k=8, step=0.04, smooth_passes=2)
        self.assertEqual(set(fx.extract(linea((0.1, 0.5), (0.9, 0.5), 60))), {0})

    def test_suavizado_quita_el_temblor(self):
        # Zigzag de ±0.004 alrededor de una recta horizontal: con suavizado queda "→" puro.
        pts = [(0.1 + 0.8 * i / 200, 0.5 + (0.004 if i % 2 else -0.004)) for i in range(201)]
        con = FeatureExtractor(k=8, step=0.04, smooth_passes=1).extract(pts)
        self.assertEqual(set(con), {0})

    def test_join_strokes_y_acepta_trazos_unidos(self):
        unido = FeatureExtractor.join_strokes([[(0, 0), (1, 1)], [(2, 2)]])
        self.assertEqual(unido, [(0, 0), (1, 1), (2, 2)])
        sec = self.fx.extract(self.fx.join_strokes([linea((0.2, 0.8), (0.2, 0.2), 30),
                                                    linea((0.2, 0.2), (0.8, 0.2), 30)]))
        self.assertEqual((sec[0], sec[-1]), (2, 0))    # sube y luego va a la derecha

    def test_extract_trazos_un_solo_trazo_es_extract(self):
        pts = linea((0.2, 0.2), (0.8, 0.7), 50)
        self.assertEqual(self.fx.extract_trazos([pts]), self.fx.extract(pts))

    def test_extract_trazos_no_depende_del_orden(self):
        # «ñ»: cuerpo (largo) y virgulilla (corta, arriba). Da igual cuál se dibuje primero.
        cuerpo = linea((0.4, 0.42), (0.4, 0.7), 40) + linea((0.4, 0.45), (0.6, 0.7), 40)[1:]
        tilde = linea((0.42, 0.3), (0.58, 0.32), 20)
        self.assertEqual(self.fx.extract_trazos([cuerpo, tilde]), self.fx.extract_trazos([tilde, cuerpo]))
        # Dos trazos principales: se ordenan de izquierda a derecha por su inicio.
        izq = linea((0.3, 0.3), (0.6, 0.7), 40)
        der = linea((0.6, 0.3), (0.3, 0.7), 40)
        self.assertEqual(self.fx.extract_trazos([izq, der]), self.fx.extract_trazos([der, izq]))

    def test_extract_trazos_salto_es_un_solo_simbolo(self):
        a = linea((0.2, 0.8), (0.2, 0.4), 30)        # sube
        b = linea((0.8, 0.8), (0.8, 0.4), 30)        # otro palito, lejos a la derecha
        sec, saltos = self.fx.extract_trazos([a, b], con_saltos=True)
        self.assertEqual(len(sec), len(self.fx.extract(a)) + 1 + len(self.fx.extract(b)))
        self.assertEqual(saltos, [len(self.fx.extract(a))])
        self.assertEqual(sec[saltos[0]], 7)           # del final de «a» (arriba) al inicio de «b» (abajo a la derecha)

    def test_extract_2d_es_alias(self):
        pts = linea((0.1, 0.1), (0.9, 0.9), 40)
        self.assertEqual(self.fx.extract_2d(pts), self.fx.extract(pts, "2d"))


class TestReconstruccion(unittest.TestCase):
    def test_ida_y_vuelta(self):
        for k in (8, 16):
            fx = FeatureExtractor(k=k, step=0.03, smooth_passes=0)
            original = [s % k for s in (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 0, 0, 3 * k // 8)]
            pts = fx.symbols_to_points(original, start=(0.3, 0.7))
            self.assertEqual(fx.extract(pts), original)

    def test_proyeccion_hacia_arriba(self):
        fx = FeatureExtractor()
        pts = fx.symbols_to_points([2, 2, 2], start=(0.5, 0.5))
        self.assertEqual(len(pts), 4)
        self.assertLess(pts[-1][1], pts[0][1])      # y de pantalla disminuye
        self.assertAlmostEqual(pts[-1][0], 0.5)

    def test_sin_simbolos_devuelve_solo_el_inicio(self):
        self.assertEqual(FeatureExtractor().symbols_to_points([], (0.2, 0.3)), [(0.2, 0.3)])

    def test_paso_personalizado(self):
        pts = FeatureExtractor().symbols_to_points([0, 0], (0.0, 0.0), step=0.1)
        self.assertAlmostEqual(pts[-1][0], 0.2)

    def test_simbolo_fuera_de_rango(self):
        fx = FeatureExtractor(k=8)
        for malo in (8, -1, 3.5, True, float("nan")):
            with self.assertRaises(ValueError, msg=f"simbolo={malo!r}"):
                fx.symbols_to_points([0, malo], (0.5, 0.5))


class TestValidaciones(unittest.TestCase):
    def test_k_solo_8_o_16(self):
        FeatureExtractor(k=8)
        FeatureExtractor(k=16)
        for k_malo in (4, 7, 9, 12, 32, 0, -8, 8.5):
            with self.assertRaises(ValueError, msg=f"k={k_malo}"):
                FeatureExtractor(k=k_malo)

    def test_parametros_invalidos(self):
        for malo in (0, -0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError, msg=f"step={malo}"):
                FeatureExtractor(step=malo)
        for malo in (-1, 1.5, "1"):
            with self.assertRaises(ValueError, msg=f"smooth={malo!r}"):
                FeatureExtractor(smooth_passes=malo)

    def test_modo_desconocido(self):
        fx = FeatureExtractor()
        pts = [(0.1, 0.1), (0.9, 0.9)]
        for mode in ("4d", "2D", "", None, "plano"):
            with self.assertRaises(ValueError, msg=f"mode={mode!r}"):
                fx.extract(pts, mode)
            with self.assertRaises(ValueError, msg=f"mode={mode!r}"):
                fx.preprocess(pts, mode)
            with self.assertRaises(ValueError, msg=f"mode={mode!r}"):
                fx.symbols_to_points([0], (0.5, 0.5), mode)

    def test_3d_aun_no_implementado(self):
        with self.assertRaises(NotImplementedError):
            FeatureExtractor().extract([(0, 0, 1), (0.1, 0, 1)], "3d")

    def test_puntos_no_finitos(self):
        fx = FeatureExtractor()
        base = linea((0.1, 0.5), (0.9, 0.5), 20)
        for malo in (float("nan"), float("inf"), float("-inf")):
            for pos in (0, 10, -1):
                pts = list(base)
                pts[pos] = (malo, 0.5)
                with self.assertRaises(ValueError):
                    fx.extract(pts)
            pts = list(base)
            pts[5] = (0.5, malo)
            with self.assertRaises(ValueError):
                fx.extract(pts)

    def test_dimension_incorrecta_y_valores_no_numericos(self):
        fx = FeatureExtractor()
        for pts in ([(0.1, 0.1, 0.1), (0.9, 0.9, 0.9)], [(0.1,), (0.9,)],
                    [("a", 0.1), (0.9, 0.9)], [None, (0.9, 0.9)]):
            with self.assertRaises(ValueError):
                fx.extract(pts)

    def test_inicio_invalido_en_proyeccion(self):
        fx = FeatureExtractor()
        for start in ((float("nan"), 0.5), (0.5, float("inf")), (0.5,), (0.1, 0.2, 0.3)):
            with self.assertRaises(ValueError):
                fx.symbols_to_points([0, 1], start)
        with self.assertRaises(ValueError):
            fx.symbols_to_points([0], (0.5, 0.5), step=0)


if __name__ == "__main__":
    unittest.main()
