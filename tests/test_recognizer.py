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
        self.assertTrue(r.proyeccion, "debería proyectar lo que falta de la letra")
        self.assertEqual(tuple(r.proyeccion[0][0]), tuple(trazos[-1][-1]))

    def test_proyeccion_levanta_el_lapiz_entre_trazos(self):
        # «i» con solo el palito: lo que falta incluye el punto, en un tramo aparte.
        muestras = [m for m in self.datos["i"] if len(m) == 2]
        palito = max(muestras[0], key=len)
        r = self.rec.procesar([palito], dibujando=True)
        if r.estado == "reconocido" and r.letra == "i":
            self.assertGreaterEqual(len(r.proyeccion), 1)
        for tramo in r.proyeccion:
            self.assertGreaterEqual(len(tramo), 2)
            self.assertTrue(all(len(p) == 2 for p in tramo))

    def test_plantillas_para_las_27_letras(self):
        self.assertEqual(len(self.rec.plantillas), 27)
        for letra, pl in self.rec.plantillas.items():
            self.assertEqual(len(pl.simbolos), len(pl.segmentos), letra)
            self.assertEqual(len(pl.ruta), len(pl.simbolos), letra)

    def test_mayoria_de_letras_completas_no_son_no_definidas(self):
        marcadas = total = 0
        for letra in ("a", "o", "s", "m"):
            for muestra in self.datos[letra][:10]:
                self.rec.reiniciar()
                total += 1
                marcadas += self.rec.procesar(muestra, dibujando=False).estado == "no_definido"
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
        r = self.rec.procesar([pts], dibujando=False)          # se decide al levantar el lápiz
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

    def test_letra_completa_no_se_confunde_con_el_comienzo_de_otra(self):
        import numpy as np
        from src.gui.recognizer import PESO_FINAL
        # Una «i» completa es el comienzo de letras más largas (la «f», la «l»): al
        # levantar el lápiz, pedir que el modelo termine en su último estado nunca debe
        # acertar menos que el prefijo, y la mayoría de las «i» deben reconocerse.
        aciertos_prefijo = aciertos_completa = total = 0
        for trazos in self.datos["i"][:15]:
            sim = self.rec.extractor.extract_trazos(trazos)
            if len(sim) < 4:
                continue
            total += 1
            aciertos_prefijo += self.rec.motor.classify(np.array(sim))[0] == "i"
            aciertos_completa += self.rec.motor.classify(np.array(sim), completa=True,
                                                         peso_final=PESO_FINAL)[0] == "i"
        self.assertGreaterEqual(aciertos_completa, aciertos_prefijo)
        self.assertGreaterEqual(aciertos_completa / total, 0.8)

    def test_peso_final_valida_el_rango(self):
        import numpy as np
        with self.assertRaises(ValueError):
            self.rec.motor.classify(np.array([6, 6, 6, 6]), completa=True, peso_final=1.5)

    def test_no_definido_no_aparece_mientras_se_dibuja(self):
        random.seed(3)
        x, y, ang, pts = 0.4, 0.5, 0.0, []
        for _ in range(200):
            if random.random() < 0.1:
                ang += random.uniform(2.4, 3.4)
            x = min(0.9, max(0.1, x + 0.007 * math.cos(ang)))
            y = min(0.9, max(0.1, y + 0.007 * math.sin(ang)))
            pts.append((x, y))
        self.assertEqual(self.rec.procesar([pts], dibujando=True).estado, "reconocido")

    def test_virgulilla_primero_no_cuenta_hasta_levantar_el_lapiz(self):
        muestra = next(m for m in self.datos["ñ"] if len(m) == 2)
        cuerpo, tilde = self.rec.extractor.ordenar_trazos(muestra)
        a_medias = cuerpo[: len(cuerpo) // 2]
        con_tilde = self.rec.procesar([tilde, a_medias], dibujando=True)
        self.rec.reiniciar()
        sin_tilde = self.rec.procesar([a_medias], dibujando=True)
        self.assertEqual(con_tilde.simbolos, sin_tilde.simbolos)
        self.rec.reiniciar()
        terminada = self.rec.procesar([tilde, cuerpo], dibujando=False)
        self.assertEqual(terminada.simbolos, self.rec.extractor.extract_trazos([cuerpo, tilde]))

    def test_proyeccion_se_ajusta_a_la_letra_del_usuario(self):
        """Letras más grandes o chicas y corridas, dibujadas a medias: lo que falta debe caer
        sobre lo que de verdad faltaba (cerca de SU línea base), y en promedio mejor que
        pegando la plantilla al cursor con la escala estimada."""
        from src.gui.guia_trazos import trazos_de_letra

        def distancia(pr, verdad):
            P = [p for t in pr for p in t]
            V = [p for t in verdad for p in t]
            return (sum(min(math.dist(p, v) for v in V) for p in P) / len(P)
                    + sum(min(math.dist(v, p) for p in P) for v in V) / len(V)) / 2

        errores = {"ajustada": [], "pegada": []}
        for letra, s, dx, dy, frac in (("n", 1.2, -0.06, 0.02, 0.5), ("h", 0.85, 0.05, 0.0, 0.55),
                                       ("a", 1.25, 0.0, -0.03, 0.45), ("m", 0.85, 0.0, 0.03, 0.4),
                                       ("y", 1.15, -0.03, 0.0, 0.6), ("d", 1.1, -0.04, 0.0, 0.5)):
            letra_entera = [[(0.5 + (x - 0.5) * s + dx, 0.56 + (y - 0.56) * s + dy) for x, y in t]
                            for t in trazos_de_letra(letra)]
            puntos = [p for t in letra_entera for p in t]
            corte = int(len(puntos) * frac)
            dibujado, verdad, cuenta = [], [], 0
            for t in letra_entera:
                if cuenta + len(t) <= corte:
                    dibujado.append(t)
                elif cuenta < corte:
                    dibujado.append(t[: corte - cuenta])
                    verdad.append(t[corte - cuenta:])
                else:
                    verdad.append(t)
                cuenta += len(t)
            sim = self.rec.extractor.extract_trazos(dibujado)
            ajustada = self.rec._proyectar(sim, letra, dibujado[-1][-1], self.rec._segmentos(dibujado)[1])
            pegada = self.rec._proyectar(sim, letra, dibujado[-1][-1])
            self.assertLess(math.dist(ajustada[0][0], dibujado[-1][-1]), 1e-9, letra)   # arranca en el cursor
            errores["ajustada"].append(distancia(ajustada, verdad))
            errores["pegada"].append(distancia(pegada, verdad))
        media = {k: sum(v) / len(v) for k, v in errores.items()}
        self.assertLess(media["ajustada"], 0.03)
        self.assertLessEqual(media["ajustada"], media["pegada"])

    def test_umbral_viene_de_los_datos(self):
        self.assertLess(self.rec.umbral, 0)
        self.assertGreater(self.rec.umbral, -10)


if __name__ == "__main__":
    unittest.main()
