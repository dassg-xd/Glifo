"""
Pruebas del reconocedor del modo 3D (src/gui/recognizer_3d.py).

Ejecutar desde la raíz del proyecto:
    python -m unittest discover -s tests -v
"""
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.FeatureExtractor import ANGULO_LIENZO, angulo_entre, pegar_en_esfera  # noqa: E402
from src.gui.recognizer_3d import Reconocedor3D  # noqa: E402

DATASET = sorted((ROOT / "data").glob("glifo-dataset-*.json"))[-1]


def rot(eje: str, a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])}[eje]


def a_cuerpo(trazos_vista, R):
    """Puntos de la vista -> marco de la esfera para una rotación R (cuerpo = Rᵀ · vista)."""
    return [[tuple(R.T @ np.asarray(p)) for p in t] for t in trazos_vista]


class TestReconocedor3D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = Reconocedor3D()
        with open(DATASET, encoding="utf-8") as f:
            cls.datos = json.load(f)

    def muestra(self, letra, i=3):
        return [[(p[0] / 1000, p[1] / 1000) for p in t] for t in self.datos[letra][i] if t]

    def test_plantillas_para_las_27(self):
        self.assertEqual(len(self.r.plantillas), 27)

    def test_el_giro_de_la_esfera_no_cambia_el_resultado(self):
        """La misma letra vista de frente da lo mismo con cualquier orientación de la esfera."""
        vista = pegar_en_esfera(self.muestra("e"), 0.1, 1.5, ANGULO_LIENZO)
        resultados = set()
        for R in (np.eye(3), rot("y", 0.7), rot("x", -0.5) @ rot("y", 2.0)):
            r = self.r.procesar_3d(a_cuerpo(vista, R), R, dibujando=False)
            resultados.add((r.letra, tuple(r.simbolos)))
        self.assertEqual(len(resultados), 1)
        self.assertEqual(resultados.pop()[0], "e")

    def test_lo_de_atras_se_ignora(self):
        vista = pegar_en_esfera(self.muestra("o"), 0.0, math.pi / 2, ANGULO_LIENZO)
        detras = a_cuerpo(vista, np.eye(3))
        self.assertEqual(self.r.procesar_3d(detras, rot("y", math.pi), dibujando=False).estado, "pocos")

    def test_zoom(self):
        """Con zoom 2, una letra de la mitad del ángulo se lee igual."""
        r = self.r.procesar_3d(pegar_en_esfera(self.muestra("a"), 0, math.pi / 2, ANGULO_LIENZO / 2),
                               np.eye(3), dibujando=False, zoom=2.0)
        self.assertEqual(r.letra, "a")

    def test_proyeccion_parte_del_cursor_y_queda_en_la_esfera(self):
        R = rot("y", 0.4)
        vista = pegar_en_esfera(self.muestra("e"), 0.1, 1.5, ANGULO_LIENZO)
        cuerpo = a_cuerpo(vista, R)
        mitad = [cuerpo[0][: len(cuerpo[0]) // 2]]
        r = self.r.procesar_3d(mitad, R, dibujando=True)
        self.assertEqual(r.estado, "reconocido")
        self.assertTrue(r.proyeccion)
        self.assertLess(angulo_entre(r.proyeccion[0][0], mitad[0][-1]), 1e-9)
        for tramo in r.proyeccion:
            for p in tramo:
                self.assertAlmostEqual(float(np.linalg.norm(p)), 1.0, places=9)

    def test_virgulilla_primero_no_cuenta_hasta_levantar_el_lapiz(self):
        muestra = next(m for m in self.datos["ñ"] if len(m) == 2)
        trazos = [[(p[0] / 1000, p[1] / 1000) for p in t] for t in muestra]
        cuerpo, tilde = self.r.extractor.ordenar_trazos(trazos)
        cuerpo3, tilde3 = pegar_en_esfera([cuerpo, tilde], 0.0, math.pi / 2, ANGULO_LIENZO)
        a_medias = cuerpo3[: len(cuerpo3) // 2]
        con_tilde = self.r.procesar_3d([tilde3, a_medias], np.eye(3), dibujando=True)
        self.r.reiniciar()
        sin_tilde = self.r.procesar_3d([a_medias], np.eye(3), dibujando=True)
        self.assertEqual(con_tilde.simbolos, sin_tilde.simbolos)

    def test_proyeccion_se_ajusta_a_la_letra_del_usuario(self):
        from src.gui.guia_trazos import trazos_de_letra
        n = [[(0.5 + (x - 0.5) * 1.15 + 0.04, 0.56 + (y - 0.56) * 1.15) for x, y in t] for t in trazos_de_letra("n")]
        vista = pegar_en_esfera(n, 0.1, math.pi / 2 - 0.05, ANGULO_LIENZO)
        mitad = [vista[0][: len(vista[0]) // 2]]
        sim = self.r.extractor.extract_trazos(mitad, "3d")
        pr = self.r._proyectar(sim, "n", mitad[-1][-1], self.r._segmentos_3d(mitad))
        self.assertLess(angulo_entre(pr[-1][-1], vista[-1][-1]), 0.06)
        for tramo in pr:
            for p in tramo:
                self.assertGreater(p[2], 0.0)                      # todo en la cara visible

    def test_vacio(self):
        self.assertEqual(self.r.procesar_3d([], np.eye(3)).estado, "vacio")


if __name__ == "__main__":
    unittest.main()
