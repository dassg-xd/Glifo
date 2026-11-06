"""
Pruebas del lector del dataset y de build_sequences.

Ejecutar desde la raíz del proyecto:
    python -m unittest discover -s tests -v
"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

import build_sequences  # noqa: E402
import dataset  # noqa: E402
from src.FeatureExtractor import ALPHABET, FeatureExtractor  # noqa: E402


def trazo_px(p0, p1, n=40, t0=0):
    """Trazo recto en la escala 0–1000 de Glifo Capture, con formato [x, y, t_ms]."""
    return [[p0[0] + (p1[0] - p0[0]) * i / n, p0[1] + (p1[1] - p0[1]) * i / n, t0 + 7 * i] for i in range(n + 1)]


def escribir(tmp, contenido):
    ruta = Path(tmp) / "glifo-dataset-test.json"
    ruta.write_text(json.dumps(contenido), encoding="utf-8")
    return ruta


class TestLoadDataset(unittest.TestCase):
    def test_normaliza_por_el_lienzo_y_descarta_el_tiempo(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = escribir(tmp, {"l": [[[[500, 250, 0], [500, 750, 99]]]]})
            datos = dataset.load_dataset(ruta, canvas=1000)
            self.assertEqual(datos["l"], [[[(0.5, 0.25), (0.5, 0.75)]]])
            datos2 = dataset.load_dataset(ruta, canvas=500)
            self.assertEqual(datos2["l"][0][0][0], (1.0, 0.5))

    def test_descarta_trazos_vacios_y_muestras_sin_puntos(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = escribir(tmp, {"a": [[[], [[1, 2, 0], [3, 4, 5]]], [[]], []]})
            datos = dataset.load_dataset(ruta)
            self.assertEqual(len(datos["a"]), 1)
            self.assertEqual(len(datos["a"][0]), 1)

    def test_errores_con_ubicacion(self):
        casos = {
            "letra desconocida": {"1": []},
            "no es objeto": [1, 2],
            "punto corto": {"a": [[[[1.0]]]]},
            "punto no numérico": {"a": [[[["x", 2, 0]]]]},
            "valor no finito": {"a": [[[[float("nan"), 2, 0]]]]},
            "letra sin lista": {"a": 5},
        }
        for nombre, contenido in casos.items():
            with tempfile.TemporaryDirectory() as tmp:
                ruta = Path(tmp) / "x.json"
                ruta.write_text(json.dumps(contenido), encoding="utf-8")   # NaN se escribe como NaN
                with self.assertRaises(ValueError, msg=nombre):
                    dataset.load_dataset(ruta)

    def test_el_error_indica_donde_esta_el_problema(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = escribir(tmp, {"a": [[[[1, 2, 0]]], [[[1, 2, 0], [3]]]]})
            with self.assertRaisesRegex(ValueError, "letra 'a', muestra 1, trazo 0, punto 1"):
                dataset.load_dataset(ruta)

    def test_canvas_invalido(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = escribir(tmp, {})
            for malo in (0, -5, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    dataset.load_dataset(ruta, malo)

    def test_find_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                dataset.find_dataset(Path(tmp))
            (Path(tmp) / "glifo-dataset-2026-10-30.json").write_text("{}")
            (Path(tmp) / "glifo-dataset-2026-11-01.json").write_text("{}")
            self.assertEqual(dataset.find_dataset(Path(tmp)).name, "glifo-dataset-2026-11-01.json")

    def test_count_by_letter_incluye_las_27(self):
        conteo = dataset.count_by_letter({"a": [1, 2], "ñ": [1]})
        self.assertEqual(len(conteo), 27)
        self.assertEqual((conteo["a"], conteo["ñ"], conteo["z"]), (2, 1, 0))


class TestBuildSequences(unittest.TestCase):
    def setUp(self):
        self.fx = FeatureExtractor(k=8, step=0.03, smooth_passes=0)

    def datos(self, n=10):
        """'l' = sube y gira a la derecha, 'v' = baja y sube, en dos trazos / uno."""
        l = [[[(x / 1000, y / 1000) for x, y, _ in trazo_px((300, 800), (300, 200))],
              [(x / 1000, y / 1000) for x, y, _ in trazo_px((300, 200), (800, 200))]] for _ in range(n)]
        v = [[[(x / 1000, y / 1000) for x, y, _ in trazo_px((200, 200), (500, 800))
               + trazo_px((500, 800), (800, 200))]] for _ in range(n)]
        return {"l": l, "v": v}

    def test_to_sequences(self):
        seqs, desc = build_sequences.to_sequences(self.datos(), self.fx, min_symbols=4)
        self.assertEqual(desc, 0)
        self.assertEqual(len(seqs["l"]), 10)
        self.assertEqual(seqs["l"][0][0], 2)       # empieza subiendo
        self.assertEqual(seqs["l"][0][-1], 0)      # termina a la derecha
        self.assertEqual(seqs["a"], [])            # las 27 letras siempre aparecen
        self.assertEqual(len(seqs), 27)

    def test_descarta_secuencias_cortas(self):
        corto = {"i": [[[(0.5, 0.5), (0.5, 0.52)]]]}
        seqs, desc = build_sequences.to_sequences(corto, self.fx, min_symbols=4)
        self.assertEqual((desc, seqs["i"]), (1, []))

    def test_split_holdout_reproducible_y_sin_fuga(self):
        seqs, _ = build_sequences.to_sequences(self.datos(20), self.fx, 4)
        for i, s in enumerate(seqs["l"]):          # etiqueta única por secuencia
            s.append(100 + i)
        tr1, te1 = build_sequences.split_holdout(seqs, 0.25, seed=1)
        tr2, te2 = build_sequences.split_holdout(seqs, 0.25, seed=1)
        self.assertEqual((tr1, te1), (tr2, te2))   # misma semilla -> mismo reparto
        self.assertEqual(len(te1["l"]), 5)
        self.assertEqual(len(tr1["l"]) + len(te1["l"]), 20)
        ids_tr = {s[-1] for s in tr1["l"]}
        ids_te = {s[-1] for s in te1["l"]}
        self.assertFalse(ids_tr & ids_te)          # ninguna muestra en ambos lados
        _, te3 = build_sequences.split_holdout(seqs, 0.25, seed=2)
        self.assertNotEqual(te1["l"], te3["l"])    # otra semilla -> otro reparto

    def test_holdout_cero(self):
        seqs, _ = build_sequences.to_sequences(self.datos(), self.fx, 4)
        train, test = build_sequences.split_holdout(seqs, 0.0, seed=1)
        self.assertEqual(len(train["l"]), 10)
        self.assertEqual(test["l"], [])

    def test_script_completo_genera_el_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            crudo = {"l": [[trazo_px((300, 800), (300, 200)), trazo_px((300, 200), (800, 200))]
                           for _ in range(10)]}
            ruta = escribir(tmp, crudo)
            salida = Path(tmp) / "sec.json"
            argv = sys.argv
            sys.argv = ["build_sequences.py", "--dataset", str(ruta), "--out", str(salida),
                        "--holdout", "0.2", "--smooth", "0"]
            try:
                self.assertEqual(build_sequences.main(), 0)
            finally:
                sys.argv = argv
            r = json.loads(salida.read_text(encoding="utf-8"))
            self.assertEqual((r["modo"], r["k"], r["n_simbolos"]), ("2d", 8, 8))
            self.assertEqual(len(r["secuencias"]["l"]), 8)
            self.assertEqual(len(r["prueba"]["l"]), 2)
            self.assertEqual(set(r["secuencias"]), set(ALPHABET))
            todos = {s for letra in r["secuencias"].values() for seq in letra for s in seq}
            self.assertTrue(todos <= set(range(8)))
            for clave in ("paso", "suavizado", "canvas", "holdout", "semilla"):
                self.assertIn(clave, r)


class TestDatasetReal(unittest.TestCase):
    """Se ejecuta solo si el JSON real del equipo está en data/."""

    def test_dataset_real(self):
        try:
            ruta = dataset.find_dataset()
        except FileNotFoundError:
            self.skipTest("No hay glifo-dataset-*.json en data/")
        datos = dataset.load_dataset(ruta)
        conteo = dataset.count_by_letter(datos)
        self.assertEqual(set(datos) | set(ALPHABET), set(ALPHABET))
        self.assertTrue(all(n > 0 for n in conteo.values()), "hay letras sin muestras")
        fx = FeatureExtractor()
        for letra in ("a", "l", "o"):
            for trazos in datos[letra][:5]:
                sec = fx.extract_trazos(trazos)
                self.assertGreater(len(sec), 3)
                self.assertTrue(all(0 <= s < 8 for s in sec))
                self.assertTrue(all(math.isfinite(x) and math.isfinite(y) for t in trazos for x, y in t))


if __name__ == "__main__":
    unittest.main()
