"""
recognizer.py
=============
Puente entre la interfaz y los módulos matemáticos.

Recibe los trazos del lienzo (puntos normalizados x/L, y/L) y devuelve lo que
la ventana 2D necesita mostrar:

1. FeatureExtractor.extract_trazos: trazos -> símbolos O_1:t (chain code,
   K = 8, orden canónico de trazos y un símbolo por salto del lápiz).
2. HMMEngine.classify: P(c | O_1:t) para las 27 letras, con escalado de
   temperatura (T = 3). Mientras el lápiz está abajo se usa la probabilidad de
   ser el comienzo de cada letra; al levantarlo, la de que la letra termine ahí.
3. Proyección con la trayectoria aprendida (ver `Plantilla`): la muestra de
   entrenamiento que el modelo de la letra ganadora considera más típica se
   alinea con el trazo del usuario por Viterbi, y se dibuja lo que le falta
   desde el cursor, con los saltos del lápiz como espacios.
4. «Trazo no definido»: si la verosimilitud por símbolo de la letra ganadora,
   log P(O_1:t | λ_ĉ) / t, queda por debajo del umbral, el trazo no se parece a
   ninguna letra. La probabilidad a posteriori no sirve para esto: al
   normalizar entre 27 modelos, hasta un garabato le da casi 100 % a alguna.
   El umbral es un percentil bajo de esa misma medida sobre las secuencias de
   entrenamiento (hoy el 3: detecta ~97 % de garabatos y marca ~3 % de letras
   reales). Si se reentrenan los modelos, hay que volver a revisarlo.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.FeatureExtractor import FeatureExtractor
from src.HMMEngine import HMMEngine

RAIZ = Path(__file__).resolve().parents[2]
RUTA_MODELOS = RAIZ / "data" / "modelos_letras_entrenados.json"
RUTA_SECUENCIAS = RAIZ / "data" / "secuencias_2d_k8.json"
CARPETA_DATOS = RAIZ / "data"
ESCALA_DATASET = 1000.0   # Glifo Capture guarda x, y en la escala 0-1000 del lienzo

MIN_SIMBOLOS = 4          # con menos, la clasificación no es fiable (contrato del extractor)
MIN_SIMBOLOS_JUICIO = 10  # no se declara «no definido» con trazos cortos (letras a medio dibujar)
PERCENTIL_UMBRAL = 3
# Escalado de temperatura de P(c | O_1:t). Calibrado con el 20 % de prueba: con
# T = 3 la confianza media mostrada (63 %) se acerca al acierto real (68 %);
# con T = 1 el programa mostraba 81 % de confianza media y muchos 100 % errados.
TEMPERATURA = 3.0
UMBRAL_POR_DEFECTO = -1.6
ESCALA_MIN, ESCALA_MAX = 0.75, 1.33
LARGO_SECUNDARIO = 0.25   # largo (fracción del lienzo) bajo el cual un trazo terminado espera al lápiz arriba


@dataclass
class Resultado:
    estado: str                      # "vacio" | "pocos" | "reconocido" | "no_definido"
    simbolos: list = field(default_factory=list)
    probabilidades: list = field(default_factory=list)   # [(letra, p)] de mayor a menor
    letra: str | None = None
    proyeccion: list = field(default_factory=list)       # [[(x, y), ...], ...] tramos; el 1.º arranca en el cursor
    verosimilitud: float | None = None                   # log P / t de la letra ganadora


@dataclass
class Plantilla:
    """Trayectoria aprendida de una letra.

    Es la muestra del dataset con mayor verosimilitud por símbolo bajo el modelo
    de su letra (la más típica según λ_c), entre las de largo intermedio. Por
    cada símbolo guarda el segmento real que lo produjo y si fue un salto del
    lápiz, y la ruta de estados de Viterbi que la alinea con el modelo.
    """
    simbolos: list
    segmentos: list          # [((x0, y0), (x1, y1), es_salto)], uno por símbolo
    ruta: np.ndarray         # estado de cada símbolo (Viterbi, terminando en el último estado)


class Reconocedor:
    def __init__(self, ruta_modelos: Path = RUTA_MODELOS, ruta_secuencias: Path | None = RUTA_SECUENCIAS,
                 carpeta_datos: Path | None = CARPETA_DATOS):
        self.motor = HMMEngine()
        self.motor.cargar_modelos(str(ruta_modelos))
        self.extractor = FeatureExtractor()
        self.umbral = self._calcular_umbral(ruta_secuencias)
        self.plantillas = self._construir_plantillas(carpeta_datos)
        self._cache_clave: tuple | None = None
        self._cache: Resultado | None = None

    # ------------------------------------------------------------------ util
    def verosimilitud_por_simbolo(self, simbolos, letra: str) -> float:
        m = self.motor.modelos[letra]
        _, c = self.motor._algoritmo_forward_escalado(m["pi"], m["A"], m["B"], np.asarray(simbolos, dtype=int))
        return float(-np.sum(np.log(c + 1e-300)) / len(simbolos))

    def _calcular_umbral(self, ruta) -> float:
        if not ruta or not Path(ruta).exists():
            return UMBRAL_POR_DEFECTO
        with open(ruta, encoding="utf-8") as f:
            datos = json.load(f)
        valores = [
            self.verosimilitud_por_simbolo(O, letra)
            for letra, seqs in datos.get("secuencias", {}).items()
            if letra in self.motor.modelos
            for O in seqs if len(O) >= MIN_SIMBOLOS
        ]
        return float(np.percentile(valores, PERCENTIL_UMBRAL)) if valores else UMBRAL_POR_DEFECTO

    def reiniciar(self) -> None:
        self._cache_clave = None
        self._cache = None

    # ------------------------------------------------------- plantillas
    def _segmentos(self, trazos) -> tuple[list, list]:
        """Símbolos y, por cada uno, el segmento que lo produjo (misma lógica que extract_trazos)."""
        fx = self.extractor
        ordenados = fx.ordenar_trazos(trazos)
        segmentos = []
        for i, t in enumerate(ordenados):
            if i > 0 and math.dist(ordenados[i - 1][-1], t[0]) > 1e-9:
                segmentos.append((tuple(ordenados[i - 1][-1]), tuple(t[0]), True))
            puntos = fx.preprocess(t)
            segmentos.extend((puntos[j], puntos[j + 1], False) for j in range(len(puntos) - 1))
        return fx.extract_trazos(trazos), segmentos

    def _construir_plantillas(self, carpeta) -> dict:
        if not carpeta:
            return {}
        candidatos = sorted(Path(carpeta).glob("glifo-dataset-*.json"))
        if not candidatos:
            return {}
        with open(candidatos[-1], encoding="utf-8") as f:
            crudo = json.load(f)
        plantillas = {}
        for letra, muestras in crudo.items():
            if letra not in self.motor.modelos:
                continue
            modelo = self.motor.modelos[letra]
            opciones = []
            for muestra in muestras:
                trazos = [[(p[0] / ESCALA_DATASET, p[1] / ESCALA_DATASET) for p in t] for t in muestra if t]
                if not trazos:
                    continue
                simbolos, segmentos = self._segmentos(trazos)
                if len(simbolos) < MIN_SIMBOLOS or len(simbolos) != len(segmentos):
                    continue
                opciones.append((simbolos, segmentos))
            if not opciones:
                continue
            largos = [len(s) for s, _ in opciones]
            bajo, alto = np.percentile(largos, 25), np.percentile(largos, 75)
            intermedias = [o for o in opciones if bajo <= len(o[0]) <= alto] or opciones
            simbolos, segmentos = max(intermedias, key=lambda o: self.verosimilitud_por_simbolo(o[0], letra))
            ruta, _ = self.motor.viterbi_log(modelo["pi"], modelo["A"], modelo["B"], simbolos, terminar_en_final=True)
            plantillas[letra] = Plantilla(simbolos, segmentos, ruta)
        return plantillas

    def _proyectar(self, simbolos, letra: str, cursor) -> list:
        """Tramos [(x, y)...] con lo que le falta a la letra, desde el cursor."""
        plantilla = self.plantillas.get(letra)
        if plantilla is None:
            # Sin plantilla: proyección con los símbolos más probables del modelo.
            _, restantes = self.motor.project(np.asarray(simbolos, dtype=int), letra=letra)
            return [self.extractor.symbols_to_points(restantes, start=cursor)] if restantes else []

        m = self.motor.modelos[letra]
        ruta_usuario, _ = self.motor.viterbi_log(m["pi"], m["A"], m["B"], simbolos)
        estado = int(ruta_usuario[-1])
        en_estado = int(np.sum(np.cumprod((ruta_usuario[::-1] == estado).astype(int))))
        indices = np.nonzero(plantilla.ruta >= estado)[0]
        if len(indices) == 0:
            return []
        inicio = int(indices[0])
        fin_estado = int(np.nonzero(plantilla.ruta == estado)[0][-1]) if np.any(plantilla.ruta == estado) else inicio
        tau = min(fin_estado, inicio + en_estado - 1)       # símbolo de la plantilla alineado con el último del usuario
        restantes = plantilla.segmentos[tau + 1:]
        if not restantes:
            return []

        # Tamaño: cada símbolo es un paso fijo de remuestreo, así que el cociente entre
        # los símbolos del usuario y los de la plantilla hasta τ estima cuánto más
        # grande o pequeña está dibujando la letra.
        escala = min(ESCALA_MAX, max(ESCALA_MIN, len(simbolos) / (tau + 1)))
        ancla = plantilla.segmentos[tau][1]

        def llevar(p):
            return (cursor[0] + (p[0] - ancla[0]) * escala, cursor[1] + (p[1] - ancla[1]) * escala)

        tramos = [[tuple(cursor)]]
        for a, b, salto in restantes:
            if salto:
                tramos.append([llevar(b)])          # lápiz arriba: el siguiente tramo empieza en otro lado
            else:
                tramos[-1].append(llevar(b))
        return [t for t in tramos if len(t) > 1]

    # ------------------------------------------------------------- principal
    def procesar(self, trazos, dibujando: bool = True) -> Resultado:
        """`trazos`: lista de trazos; cada trazo, lista de puntos (x, y) normalizados.

        `dibujando`: True mientras el lápiz está abajo (la letra puede seguir);
        False al levantarlo (se evalúa como letra terminada).
        """
        trazos = [t for t in trazos if t]
        if not trazos:
            self.reiniciar()
            return Resultado("vacio")

        cursor = tuple(trazos[-1][-1])
        if dibujando and len(trazos) > 1:
            # Mientras se dibuja, los trazos que en el orden canónico irían DESPUÉS del
            # trazo en curso (p. ej. la virgulilla hecha antes que la «n») todavía no se
            # tienen en cuenta: la secuencia «n a medias + salto + virgulilla» no existe
            # en el entrenamiento. Al levantar el lápiz entran en su lugar.
            # En el dataset los trazos secundarios miden como mucho ~0.20 del lienzo y
            # los principales desde ~0.21: los trazos terminados más cortos que
            # LARGO_SECUNDARIO esperan al lápiz arriba. El trazo en curso siempre cuenta.
            def largo(t):
                return sum(math.dist(a, b) for a, b in zip(t, t[1:]))
            terminados = [t for t in trazos[:-1] if largo(t) >= LARGO_SECUNDARIO]
            # Los terminados van en orden canónico y el trazo en curso al final, que es
            # donde está el cursor y desde donde se proyecta lo que falta.
            trazos = self.extractor.ordenar_trazos(terminados) + [trazos[-1]]

        simbolos = self.extractor.extract_trazos(trazos)
        if len(simbolos) < MIN_SIMBOLOS:
            return Resultado("pocos", simbolos=simbolos)

        clave = (tuple(simbolos), dibujando)
        if clave != self._cache_clave:
            # Solo se recalcula el HMM cuando aparece un símbolo nuevo o se levanta el lápiz.
            letra, probs = self.motor.classify(np.asarray(simbolos, dtype=int), temperatura=TEMPERATURA,
                                               completa=not dibujando)
            ordenadas = sorted(((l, float(p)) for l, p in probs.items()), key=lambda x: -x[1])
            vps = self.verosimilitud_por_simbolo(simbolos, letra)
            # «No definido» solo se decide con el lápiz arriba: a mitad de un trazo,
            # una letra real poco común puede pasar un momento por debajo del umbral.
            no_definido = (not dibujando) and len(simbolos) >= MIN_SIMBOLOS_JUICIO and vps < self.umbral
            self._cache_clave = clave
            self._cache = Resultado("no_definido" if no_definido else "reconocido",
                                    simbolos=simbolos, probabilidades=ordenadas, letra=letra,
                                    verosimilitud=vps)

        r = self._cache
        proyeccion = [] if r.estado == "no_definido" else self._proyectar(r.simbolos, r.letra, cursor)
        return Resultado(r.estado, r.simbolos, r.probabilidades, r.letra, proyeccion, r.verosimilitud)
