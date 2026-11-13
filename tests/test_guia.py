"""
Pruebas de la guía de dibujo (data/guia_trazos.json, src/gui/guia_trazos.py).

La guía sale del dataset (training/generar_guia.py), así que quien la sigue debe
ser reconocido: si cambia el dataset y no se regenera la guía, estas pruebas lo
avisan.

Ejecutar desde la raíz del proyecto:
    python -m unittest discover -s tests -v
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.FeatureExtractor import ALPHABET, ANGULO_LIENZO, pegar_en_esfera  # noqa: E402
from src.gui.guia_trazos import GUIA, puntos_de_camino, segmentos_de_camino, trazos_de_letra  # noqa: E402
from src.gui.recognizer import Reconocedor  # noqa: E402
from src.gui.recognizer_3d import Reconocedor3D  # noqa: E402


class TestCaminos(unittest.TestCase):
    def test_las_27_letras(self):
        self.assertEqual(set(GUIA), set(ALPHABET))

    def test_lector_de_caminos(self):
        piezas = segmentos_de_camino("M100 200 L300 200 C350 200 400 250 400 300")
        self.assertEqual([p[0] for p in piezas], ["L", "C"])
        pts = puntos_de_camino("M100 200 L300 200", paso=10)
        self.assertEqual(pts[0], (100.0, 200.0))
        self.assertEqual(pts[-1], (300.0, 200.0))
        self.assertEqual(len(pts), 21)

    def test_dentro_del_lienzo_y_entre_las_lineas(self):
        for letra in ALPHABET:
            for t in trazos_de_letra(letra):
                for x, y in t:
                    self.assertTrue(0.1 <= x <= 0.9 and 0.1 <= y <= 0.95, (letra, x, y))


class TestQuienSigueLaGuiaEsReconocido(unittest.TestCase):
    def test_2d(self):
        r = Reconocedor()
        malas = []
        for letra in ALPHABET:
            r.reiniciar()
            res = r.procesar(trazos_de_letra(letra), dibujando=False)
            if not (res.estado == "reconocido" and res.letra == letra):
                malas.append((letra, res.estado, res.letra))
        self.assertLessEqual(len(malas), 1, malas)

    def test_3d(self):
        r = Reconocedor3D()
        malas = []
        for letra in ALPHABET:
            r.reiniciar()
            vista = pegar_en_esfera(trazos_de_letra(letra), 0.05, math.pi / 2, ANGULO_LIENZO)
            res = r.procesar_3d(vista, np.eye(3), dibujando=False)
            if not (res.estado == "reconocido" and res.letra == letra):
                malas.append((letra, res.estado, res.letra))
        self.assertLessEqual(len(malas), 1, malas)


if __name__ == "__main__":
    unittest.main()
