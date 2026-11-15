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

    def test_se_reconoce_en_cualquier_longitud_incluso_atras(self):
        """Cuenta toda la esfera: la misma letra da lo mismo adelante, de lado o en la cara de atrás."""
        resultados = set()
        for theta in (0.1, 1.6, math.pi, -2.4):
            r = self.r.procesar_3d(pegar_en_esfera(self.muestra("e"), theta, 1.5, ANGULO_LIENZO), dibujando=False)
            resultados.add((r.letra, tuple(r.simbolos)))
        self.assertEqual(len(resultados), 1)
        self.assertEqual(resultados.pop()[0], "e")

    def test_arriba_es_el_polo_norte_no_la_pantalla(self):
        """Una letra derecha respecto al polo norte se reconoce; la misma letra acostada 90°
        sobre la esfera ya no da los mismos símbolos (no existe un «arriba» de la pantalla)."""
        derecha = pegar_en_esfera(self.muestra("e"), 0.3, 1.5, ANGULO_LIENZO)
        acostada = pegar_en_esfera(self.muestra("e"), 0.3, 1.5, ANGULO_LIENZO, giro=math.pi / 2)
        r1 = self.r.procesar_3d(derecha, dibujando=False)
        self.r.reiniciar()
        r2 = self.r.procesar_3d(acostada, dibujando=False)
        self.assertEqual(r1.letra, "e")
        self.assertNotEqual(r1.simbolos, r2.simbolos)

    def test_zoom(self):
        """Con zoom 2, una letra de la mitad del ángulo se lee igual."""
        r = self.r.procesar_3d(pegar_en_esfera(self.muestra("a"), 0, math.pi / 2, ANGULO_LIENZO / 2),
                               dibujando=False, zoom=2.0)
        self.assertEqual(r.letra, "a")

    def test_proyeccion_parte_del_cursor_y_queda_en_la_esfera(self):
        for theta in (0.1, 2.9):                                   # adelante y en la cara de atrás
            cuerpo = pegar_en_esfera(self.muestra("e"), theta, 1.5, ANGULO_LIENZO)
            mitad = [cuerpo[0][: len(cuerpo[0]) // 2]]
            self.r.reiniciar()
            r = self.r.procesar_3d(mitad, dibujando=True)
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
        con_tilde = self.r.procesar_3d([tilde3, a_medias], dibujando=True)
        self.r.reiniciar()
        sin_tilde = self.r.procesar_3d([a_medias], dibujando=True)
        self.assertEqual(con_tilde.simbolos, sin_tilde.simbolos)

    def test_proyeccion_se_ajusta_a_la_letra_del_usuario(self):
        from src.gui.guia_trazos import trazos_de_letra
        n = [[(0.5 + (x - 0.5) * 1.15 + 0.04, 0.56 + (y - 0.56) * 1.15) for x, y in t] for t in trazos_de_letra("n")]
        for theta in (0.1, -2.7):                                  # adelante y en la cara de atrás
            entera = pegar_en_esfera(n, theta, math.pi / 2 - 0.05, ANGULO_LIENZO)
            mitad = [entera[0][: len(entera[0]) // 2]]
            sim = self.r.extractor.extract_trazos(mitad, "3d")
            pr = self.r._proyectar_3d(sim, "n", mitad, mitad[-1][-1])
            self.assertLess(angulo_entre(pr[-1][-1], entera[-1][-1]), 0.06)

    def test_vacio(self):
        self.assertEqual(self.r.procesar_3d([]).estado, "vacio")


if __name__ == "__main__":
    unittest.main()
