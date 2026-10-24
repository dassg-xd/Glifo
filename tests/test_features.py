"""
Pruebas de FeatureExtractor y del manejo de datos.

Ejecutar desde la raíz del proyecto:
    python -m unittest discover -s tests -v
"""
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

import dataset  # noqa: E402
from src.FeatureExtractor import (  # noqa: E402
    ALPHABET,
    FeatureExtractor,
    quantize_angle,
    symbol_to_angle,
    to_spherical,
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
        # Justo por debajo / por encima de la frontera 22.5° entre sector 0 y 1
        self.assertEqual(quantize_angle(sector / 2 - 1e-6, 8), 0)
        self.assertEqual(quantize_angle(sector / 2 + 1e-6, 8), 1)

    def test_angulos_negativos_y_mayores_a_2pi(self):
        self.assertEqual(quantize_angle(-math.pi / 2, 8), 6)       # abajo
        self.assertEqual(quantize_angle(2 * math.pi + 0.01, 8), 0)
        self.assertEqual(quantize_angle(-0.01, 8), 0)

    def test_alfabeto_tiene_27_letras(self):
        self.assertEqual(len(ALPHABET), 27)
        self.assertIn("ñ", ALPHABET)
        self.assertEqual(len(set(ALPHABET)), 27)


class TestExtract2D(unittest.TestCase):
    def setUp(self):
        self.fx = FeatureExtractor(k=8, step_2d=0.04, smooth_passes=0)

    def test_direcciones_cardinales(self):
        # Recuerda: en pantalla Y crece hacia abajo, así que "arriba" = y decreciente.
        casos = {
            0: ((0.2, 0.5), (0.8, 0.5)),   # →
            2: ((0.5, 0.8), (0.5, 0.2)),   # ↑
            4: ((0.8, 0.5), (0.2, 0.5)),   # ←
            6: ((0.5, 0.2), (0.5, 0.8)),   # ↓
            1: ((0.2, 0.8), (0.8, 0.2)),   # ↗
            5: ((0.8, 0.2), (0.2, 0.8)),   # ↙
        }
        for simbolo, (a, b) in casos.items():
            sec = self.fx.extract_2d(linea(a, b, 50))
            self.assertTrue(len(sec) > 5)
            self.assertEqual(set(sec), {simbolo}, f"dirección {simbolo}")

    def test_invariante_a_la_velocidad(self):
        lento = linea((0.1, 0.5), (0.9, 0.5), 400)   # muchos eventos del ratón
        rapido = linea((0.1, 0.5), (0.9, 0.5), 7)    # pocos eventos
        self.assertEqual(self.fx.extract_2d(lento), self.fx.extract_2d(rapido))

    def test_longitud_proporcional_al_recorrido(self):
        sec = self.fx.extract_2d(linea((0.1, 0.5), (0.9, 0.5), 100))
        self.assertAlmostEqual(len(sec), 0.8 / 0.04, delta=1)

    def test_puntos_repetidos_no_generan_simbolos(self):
        pts = [(0.5, 0.5)] * 20 + linea((0.5, 0.5), (0.9, 0.5), 30)
        self.assertEqual(set(self.fx.extract_2d(pts)), {0})

    def test_entradas_degeneradas(self):
        self.assertEqual(self.fx.extract_2d([]), [])
        self.assertEqual(self.fx.extract_2d([(0.5, 0.5)]), [])
        self.assertEqual(self.fx.extract_2d([(0.5, 0.5), (0.5, 0.5)]), [])
        self.assertEqual(self.fx.extract_2d([(0.5, 0.5), (0.51, 0.5)]), [])  # menor a un paso

    def test_circulo_antihorario_gira_los_simbolos(self):
        # Antihorario visto en pantalla (Y invertido): 0,1,2,...,7,0,...
        pts = []
        for i in range(201):
            t = 2 * math.pi * i / 200
            pts.append((0.5 + 0.25 * math.cos(t), 0.5 - 0.25 * math.sin(t)))
        sec = self.fx.extract_2d(pts)
        saltos = {(b - a) % 8 for a, b in zip(sec, sec[1:])}
        self.assertTrue(saltos <= {0, 1}, f"saltos inesperados: {saltos}")
        self.assertGreaterEqual(len(set(sec)), 7)  # recorre casi todos los sectores

    def test_suavizado_mantiene_extremos_y_forma(self):
        fx = FeatureExtractor(k=8, smooth_passes=2)
        sec = fx.extract_2d(linea((0.1, 0.5), (0.9, 0.5), 60))
        self.assertEqual(set(sec), {0})

    def test_ida_y_vuelta_2d(self):
        for k in (8, 16):
            fx = FeatureExtractor(k=k, smooth_passes=0)
            original = [s % k for s in (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 0, 0, 3 * k // 8)]
            pts = fx.symbols_to_points_2d(original, start=(0.3, 0.7))
            self.assertEqual(fx.extract_2d(pts), original)

    def test_proyeccion_hacia_arriba(self):
        fx = FeatureExtractor(k=8)
        pts = fx.symbols_to_points_2d([2, 2, 2], start=(0.5, 0.5))
        self.assertEqual(len(pts), 4)
        self.assertLess(pts[-1][1], pts[0][1])  # y de pantalla disminuye
        self.assertAlmostEqual(pts[-1][0], 0.5)

    def test_join_strokes(self):
        unido = FeatureExtractor.join_strokes([[(0, 0), (1, 1)], [(2, 2)]])
        self.assertEqual(unido, [(0, 0), (1, 1), (2, 2)])


class TestExtract3D(unittest.TestCase):
    def setUp(self):
        self.fx = FeatureExtractor(k=8, step_3d=0.08, smooth_passes=0)

    def test_esferica_en_puntos_conocidos(self):
        th, ph = to_spherical(0, 0, 1)     # centro visible: ecuador, longitud 0
        self.assertAlmostEqual(th, math.pi / 2)
        self.assertAlmostEqual(ph, 0.0)
        th, ph = to_spherical(1, 0, 0)     # derecha
        self.assertAlmostEqual(ph, math.pi / 2)
        th, _ = to_spherical(0, 1, 0)      # polo superior
        self.assertAlmostEqual(th, 0.0)
        th, _ = to_spherical(0.0, -2.0, 0.0)   # también acepta puntos no unitarios
        self.assertAlmostEqual(th, math.pi)

    def test_punto_nulo_falla(self):
        with self.assertRaises(ValueError):
            to_spherical(0, 0, 0)

    def test_ida_y_vuelta_3d(self):
        original = [0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 0]
        pts = self.fx.symbols_to_points_3d(original, start=(0.0, 0.0, 1.0))
        self.assertEqual(self.fx.extract_3d(pts), original)

    def test_ida_y_vuelta_3d_k16(self):
        fx = FeatureExtractor(k=16, step_3d=0.08, smooth_passes=0)
        original = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        pts = fx.symbols_to_points_3d(original, start=(0.0, 0.0, 1.0))
        self.assertEqual(fx.extract_3d(pts), original)

    def test_puntos_proyectados_estan_en_la_esfera(self):
        pts = self.fx.symbols_to_points_3d([0, 1, 2, 3] * 5, start=(0.3, 0.2, math.sqrt(1 - 0.13)))
        for p in pts:
            self.assertAlmostEqual(math.sqrt(sum(c * c for c in p)), 1.0, places=9)

    def test_cruce_de_phi_en_pi(self):
        # Recorrido hacia la derecha por el ecuador cruzando φ = ±π (parte trasera):
        # el salto 179° -> -179° debe contar como +2°, no como -358°.
        phi0 = math.pi - 0.2
        inicio = (math.sin(phi0), 0.0, math.cos(phi0))
        pts = self.fx.symbols_to_points_3d([0] * 8, start=inicio)
        fis = [to_spherical(*p)[1] for p in pts]
        self.assertTrue(any(f < 0 for f in fis) and any(f > 0 for f in fis), "el trazo debe cruzar ±π")
        self.assertEqual(self.fx.extract_3d(pts), [0] * 8)

    def test_invariante_a_la_densidad_de_puntos_3d(self):
        denso = self.fx.symbols_to_points_3d([0] * 10, start=(0.0, 0.0, 1.0), step=0.01)
        ralo = self.fx.symbols_to_points_3d([0] * 10, start=(0.0, 0.0, 1.0), step=0.08)
        a, b = self.fx.extract_3d(denso), self.fx.extract_3d(ralo)
        self.assertEqual(set(a), {0})
        self.assertEqual(set(b), {0})

    def test_arriba_en_3d_es_simbolo_2(self):
        pts = self.fx.symbols_to_points_3d([2] * 6, start=(0.0, 0.0, 1.0))
        self.assertGreater(pts[-1][1], pts[0][1])   # y de la cámara aumenta (sube)
        self.assertEqual(self.fx.extract_3d(pts), [2] * 6)

    def test_entradas_degeneradas_3d(self):
        self.assertEqual(self.fx.extract_3d([]), [])
        self.assertEqual(self.fx.extract_3d([(0, 0, 1)]), [])

    def test_atajos_por_modo(self):
        fx = FeatureExtractor(k=8, smooth_passes=0)
        p2 = fx.symbols_to_points([0, 0], (0.1, 0.1), mode="2d")
        p3 = fx.symbols_to_points([0, 0], (0, 0, 1), mode="3d")
        self.assertEqual(len(p2[0]), 2)
        self.assertEqual(len(p3[0]), 3)
        self.assertEqual(fx.extract(p2, "2d"), [0, 0])
        self.assertEqual(fx.extract(p3, "3d"), [0, 0])


class TestParametros(unittest.TestCase):
    def test_validaciones(self):
        with self.assertRaises(ValueError):
            FeatureExtractor(k=2)
        with self.assertRaises(ValueError):
            FeatureExtractor(step_2d=0)


class TestDataset(unittest.TestCase):
    def test_guardar_leer_contar_borrar(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            trazo = [[(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)]]
            dataset.save_sample("a", "2d", trazo, "juan", d)
            dataset.save_sample("ñ", "2d", trazo, "Ana Maria!", d)   # nombre con símbolos
            dataset.save_sample("a", "2d", trazo, "ana", d)

            self.assertTrue((d / "2d" / "enie").exists())            # ñ -> carpeta 'enie'
            total = dataset.count_samples("2d", d)
            self.assertEqual(total["a"], 2)
            self.assertEqual(total["ñ"], 1)
            self.assertEqual(dataset.count_samples("2d", d, author="juan")["a"], 1)
            self.assertEqual(sum(dataset.count_samples("3d", d).values()), 0)

            muestras = dataset.load_samples("2d", d)
            self.assertEqual(len(muestras), 3)
            self.assertEqual({m["letter"] for m in muestras}, {"a", "ñ"})

            borrado = dataset.delete_last_sample("a", "2d", "juan", d)
            self.assertIsNotNone(borrado)
            self.assertEqual(dataset.count_samples("2d", d)["a"], 1)
            self.assertIsNone(dataset.delete_last_sample("a", "2d", "juan", d))

    def test_validaciones_de_guardado(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                dataset.save_sample("1", "2d", [[(0, 0), (1, 1)]], "x", Path(tmp))
            with self.assertRaises(ValueError):
                dataset.save_sample("a", "4d", [[(0, 0), (1, 1)]], "x", Path(tmp))
            with self.assertRaises(ValueError):
                dataset.save_sample("a", "2d", [[(0, 0)]], "x", Path(tmp))

    def test_mapeo_de_pantalla(self):
        self.assertEqual(dataset.screen_to_unit(260, 130, 520), (0.5, 0.25))
        x, y, z = dataset.screen_to_sphere(260, 260, 260, 260, 240)   # centro del disco
        self.assertEqual((x, y, z), (0.0, 0.0, 1.0))
        x, y, z = dataset.screen_to_sphere(260, 20, 260, 260, 240)    # arriba del disco
        self.assertAlmostEqual(y, 1.0)
        self.assertAlmostEqual(z, 0.0)
        x, y, z = dataset.screen_to_sphere(900, 260, 260, 260, 240)   # fuera: se ajusta al borde
        self.assertAlmostEqual(math.sqrt(x * x + y * y + z * z), 1.0)
        self.assertEqual(z, 0.0)

    def test_flujo_completo_muestra_a_secuencia(self):
        """Lo que hace build_sequences: JSON guardado -> símbolos."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            fx = FeatureExtractor(k=8, smooth_passes=0)
            trazo1 = linea((0.2, 0.8), (0.2, 0.2), 30)       # sube
            trazo2 = linea((0.2, 0.2), (0.8, 0.2), 30)       # luego derecha
            dataset.save_sample("l", "2d", [trazo1, trazo2], "t", d)
            m = dataset.load_samples("2d", d)[0]
            sec = fx.extract(fx.join_strokes(m["strokes"]), "2d")
            self.assertEqual(sec[0], 2)
            self.assertEqual(sec[-1], 0)


if __name__ == "__main__":
    unittest.main()
