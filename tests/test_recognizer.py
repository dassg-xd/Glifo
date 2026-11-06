"""Pruebas del puente GUI <-> HMM (no necesitan Qt)."""
import math
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from dataset import find_dataset, load_dataset  # noqa: E402
from src.gui.recognizer import MIN_SIMBOLOS, Reconocedor  # noqa: E402


class TestReconocedor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = Reconocedor()
        cls.datos = load_dataset(find_dataset())

    def setUp(self):
        self.rec.reiniciar()

    def test_sin_trazos_es_vacio(self):
        self.assertEqual(self.rec.procesar([]).estado, "vacio")
        self.assertEqual(self.rec.procesar([[]]).estado, "vacio")

    def test_trazo_muy_corto_espera_mas_simbolos(self):
        r = self.rec.procesar([[(0.5, 0.5), (0.5, 0.52)]])
        self.assertEqual(r.estado, "pocos")
        self.assertLess(len(r.simbolos), MIN_SIMBOLOS)

    def test_letra_real_se_reconoce_y_proyecta_desde_el_cursor(self):
        muestra = self.datos["l"][0]
        trazos = [t[: len(t) // 2] for t in muestra]      # letra a medio dibujar
        r = self.rec.procesar(trazos)
        self.assertEqual(r.estado, "reconocido")
        self.assertEqual(len(r.probabilidades), 27)
        self.assertAlmostEqual(sum(p for _, p in r.probabilidades), 1.0, places=6)
        self.assertEqual(r.letra, r.probabilidades[0][0])
        if r.proyeccion:
            self.assertEqual(tuple(r.proyeccion[0]), tuple(trazos[-1][-1]))

    def test_mayoria_de_letras_completas_no_son_no_definidas(self):
        marcadas = total = 0
        for letra in ("a", "o", "s", "m"):
            for muestra in self.datos[letra][:10]:
                self.rec.reiniciar()
                total += 1
                marcadas += self.rec.procesar(muestra).estado == "no_definido"
        self.assertLessEqual(marcadas / total, 0.15)

    def test_garabato_es_no_definido(self):
        random.seed(3)
        x, y, ang, pts = 0.4, 0.5, 0.0, []
        for _ in range(200):
            if random.random() < 0.1:
                ang += random.uniform(2.4, 3.4)          # cambios bruscos: tachón
            x = min(0.9, max(0.1, x + 0.007 * math.cos(ang)))
            y = min(0.9, max(0.1, y + 0.007 * math.sin(ang)))
            pts.append((x, y))
        r = self.rec.procesar([pts])
        self.assertEqual(r.estado, "no_definido")
        self.assertEqual(r.proyeccion, [])

    def test_temperatura_suaviza_sin_cambiar_el_orden(self):
        import numpy as np
        simbolos = self.rec.extractor.extract(self.rec.extractor.join_strokes(self.datos["m"][0]))
        g1, p1 = self.rec.motor.classify(np.array(simbolos), temperatura=1.0)
        g3, p3 = self.rec.motor.classify(np.array(simbolos), temperatura=3.0)
        self.assertEqual(g1, g3)
        self.assertEqual(sorted(p1, key=p1.get), sorted(p3, key=p3.get))
        self.assertLessEqual(p3[g3], p1[g1] + 1e-12)
        self.assertAlmostEqual(sum(p3.values()), 1.0, places=6)

    def test_umbral_viene_de_los_datos(self):
        self.assertLess(self.rec.umbral, 0)
        self.assertGreater(self.rec.umbral, -10)


if __name__ == "__main__":
    unittest.main()
