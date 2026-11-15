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
RUTA_SECUENCIAS = RAIZ / "data" / "secuencias_2d_k16.json"
CARPETA_DATOS = RAIZ / "data"
ESCALA_DATASET = 1000.0   # Glifo Capture guarda x, y en la escala 0-1000 del lienzo

MIN_SIMBOLOS = 4          # con menos, la clasificación no es fiable (contrato del extractor)
MIN_SIMBOLOS_JUICIO = 10  # no se declara «no definido» con trazos cortos (letras a medio dibujar)
PERCENTIL_UMBRAL = 3
# Escalado de temperatura de P(c | O_1:t). Calibrado con el 20 % de prueba: con
# T = 3 la confianza media mostrada (63 %) se acerca al acierto real (68 %);
# con T = 1 el programa mostraba 81 % de confianza media y muchos 100 % errados.
TEMPERATURA = 3.0
# Al levantar el lápiz, cuánto pesa exigir que el modelo termine en su último estado
# (ver HMMEngine.classify). Elegido con las secuencias de ENTRENAMIENTO del dataset de un
# diseño por letra: 0.2 dio 92.9 % (1 = solo letra completa: 90.1 %; 0 = solo prefijo:
# 92.2 %). En el 20 % de prueba: 93.8 %.
PESO_FINAL = 0.2
# Al levantar el lápiz, cuánto pesa la probabilidad del largo de la secuencia
# (ver HMMEngine.classify). Elegido con las secuencias de ENTRENAMIENTO: 2 dio el mejor
# acierto en 2D y casi el mejor en 3D. En la prueba: 2D 93.8 % -> 95.8 %, 3D 91.6 % -> 93.1 %.
PESO_LARGO = 2.0
UMBRAL_POR_DEFECTO = -1.6
ESCALA_MIN, ESCALA_MAX = 0.75, 1.33
# Ajuste de la plantilla al dibujo del usuario (ver `_ajustar`): mínimo de puntos
# emparejados, rango de escala permitido y en qué fracción del alto de la letra la
# proyección pasa de arrancar en el cursor a la posición ajustada.
AJUSTE_MIN_PUNTOS = 4
DTW_ESTIRAR = 0.5
AJUSTE_ESCALA = (0.6, 1.8)
TRANSICION = 0.25
LARGO_SECUNDARIO = 0.25   # largo (fracción del lienzo) bajo el cual un trazo terminado espera al lápiz arriba


def alinear_dtw(usuario, plantilla, k: int = 8) -> tuple[int, list]:
    """DTW de final abierto entre los símbolos del usuario (una letra a medias) y los de la
    plantilla (la letra completa): cuánto de la plantilla cubre lo dibujado.

    Recibe DIRECCIONES (0..k-1, sin la zona). Costo entre dos = diferencia angular medida
    en octavos de vuelta (0 a 4, sea k = 8 o 16), más DTW_ESTIRAR
    por cada paso que no avanza en las dos secuencias a la vez (sin esa penalización, una
    racha de símbolos iguales se «apila» gratis sobre un solo símbolo de la plantilla y el
    final queda atrasado). Se alinea TODO lo del usuario con un PREFIJO de la plantilla;
    el final τ es el que minimiza el costo acumulado por paso. Devuelve (τ, camino) con el camino como pares (j_usuario, i_plantilla).

    No usa los estados de Viterbi porque estados vecinos pueden emitir lo mismo (en la «n»,
    el regreso por el palito ↑ se reparte entre dos estados) y para una letra a medias la
    ruta se queda en el primero: la alineación se atascaba."""
    u, p = list(usuario), list(plantilla)
    n, m = len(u), len(p)
    if n == 0 or m == 0:
        return 0, []
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            d = abs(u[j - 1] - p[i - 1]) % k
            costo = min(d, k - d) * 8.0 / k
            D[j, i] = costo + min(D[j - 1, i - 1], D[j - 1, i] + DTW_ESTIRAR, D[j, i - 1] + DTW_ESTIRAR)
    finales = D[n, 1:] / (n + np.arange(1, m + 1))
    tau = int(np.argmin(finales))
    camino, j, i = [], n, tau + 1
    while j > 0 and i > 0:
        camino.append((j - 1, i - 1))
        paso = np.argmin([D[j - 1, i - 1], D[j - 1, i] + DTW_ESTIRAR, D[j, i - 1] + DTW_ESTIRAR])
        if paso == 0:
            j, i = j - 1, i - 1
        elif paso == 1:
            j -= 1
        else:
            i -= 1
    return tau, camino[::-1]


def es_secundario_previo(terminado, actual) -> bool:
    """¿Un trazo ya terminado debe esperar al lápiz arriba mientras se dibuja `actual`?
    Sí si es muy corto o si queda entero por encima del trazo en curso (la y crece
    hacia abajo): así un punto o una virgulilla hechos antes que el cuerpo no se
    mezclan con la letra a medias."""
    largo = sum(math.dist(a, b) for a, b in zip(terminado, terminado[1:]))
    return largo < LARGO_SECUNDARIO or max(p[1] for p in terminado) < min(p[1] for p in actual)


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

    Es el prototipo de la guía de dibujo (el trazo mediano de todas las muestras
    de la letra) o, si falta, la muestra del dataset con mayor verosimilitud por
    símbolo bajo el modelo de su letra. Por cada símbolo guarda el segmento que
    lo produjo y si fue un salto del lápiz, y la ruta de estados de Viterbi que
    la alinea con el modelo.
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
        self.largos = self._calcular_largos(ruta_secuencias)
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

    @staticmethod
    def _calcular_largos(ruta) -> dict:
        """{letra: (media, desviación)} del largo de las secuencias de entrenamiento. A la
        desviación se le suma 1 símbolo para no castigar de más a las letras muy regulares."""
        if not ruta or not Path(ruta).exists():
            return {}
        with open(ruta, encoding="utf-8") as f:
            datos = json.load(f)
        return {letra: (float(np.mean([len(O) for O in seqs])), float(np.std([len(O) for O in seqs])) + 1.0)
                for letra, seqs in datos.get("secuencias", {}).items() if seqs}

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
        """Una plantilla por letra. Primero el prototipo de la guía de dibujo (el trazo
        mediano del dataset, limpio: data/guia_trazos.json); si falta, la muestra del
        dataset que el modelo considera más típica."""
        plantillas = {}
        try:
            from src.gui.guia_trazos import GUIA, trazos_de_letra
        except (OSError, ValueError, KeyError):
            GUIA = {}
        for letra in self.motor.modelos:
            if letra not in GUIA:
                continue
            simbolos, segmentos = self._segmentos(trazos_de_letra(letra))
            if len(simbolos) < MIN_SIMBOLOS or len(simbolos) != len(segmentos):
                continue
            modelo = self.motor.modelos[letra]
            ruta, _ = self.motor.viterbi_log(modelo["pi"], modelo["A"], modelo["B"], simbolos, terminar_en_final=True)
            plantillas[letra] = Plantilla(simbolos, segmentos, ruta)
        if not carpeta:
            return plantillas
        candidatos = sorted(Path(carpeta).glob("glifo-dataset-*.json"))
        if not candidatos:
            return plantillas
        with open(candidatos[-1], encoding="utf-8") as f:
            crudo = json.load(f)
        for letra, muestras in crudo.items():
            if letra not in self.motor.modelos or letra in plantillas:
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

    def _alinear(self, simbolos, letra: str, plantilla: Plantilla):
        """(τ, escala) o None. τ es el símbolo de la plantilla alineado con el último del
        usuario y la escala, cuánto más grande o pequeña está dibujando la letra. None si a
        la plantilla ya no le queda nada. La alineación es un DTW de final abierto (ver
        `alinear_dtw`)."""
        tau, _ = self._dtw(simbolos, letra, plantilla)
        if tau + 1 >= len(plantilla.segmentos):
            return None
        # Cada símbolo es un paso fijo de remuestreo, así que el cociente entre los
        # símbolos del usuario y los de la plantilla hasta τ estima el tamaño relativo.
        return tau, min(ESCALA_MAX, max(ESCALA_MIN, len(simbolos) / (tau + 1)))

    def _dtw(self, simbolos, letra: str, plantilla: Plantilla):
        clave = (tuple(simbolos), letra)
        if getattr(self, "_dtw_clave", None) != clave:
            d = self.extractor.direccion            # el DTW compara direcciones, sin la zona
            self._dtw_clave, self._dtw_valor = clave, alinear_dtw(
                [d(x) for x in simbolos], [d(x) for x in plantilla.simbolos], self.extractor.k)
        return self._dtw_valor

    # Coordenadas del plano en el que se ajusta y se dibuja la proyección. En 2D es el
    # lienzo; el modo 3D las cambia por lo que se ve en pantalla (ver Reconocedor3D).
    def _plano(self, p) -> tuple:
        return (float(p[0]), float(p[1]))

    def _desde_plano(self, q):
        return (float(q[0]), float(q[1]))

    def _ajustar(self, simbolos, segmentos_usuario, letra: str, plantilla: Plantilla, tau: int,
                 escala_base: float = 1.0):
        """Escala s y traslación t que mejor llevan la plantilla sobre lo que ya dibujó el
        usuario (mínimos cuadrados), o None si no hay con qué ajustar.

        Los pares de puntos (usuario, plantilla) salen del camino del DTW. Con ellos,
        s = Σ(T−T̄)·(U−Ū) / Σ|T−T̄|² y t = Ū − s·T̄. No hay giro: arriba siempre es arriba."""
        if len(segmentos_usuario) != len(simbolos):
            return None
        _, camino = self._dtw(simbolos, letra, plantilla)
        pares = [(self._plano(plantilla.segmentos[i][1]), self._plano(segmentos_usuario[j][1]))
                 for j, i in camino
                 if not segmentos_usuario[j][2] and not plantilla.segmentos[i][2]]   # los saltos no son tinta
        if len(pares) < AJUSTE_MIN_PUNTOS:
            return None
        T = np.array([a for a, _ in pares])
        U = np.array([b for _, b in pares])
        dT = T - T.mean(axis=0)
        var = float((dT ** 2).sum())
        if var < 1e-9:
            return None
        s = float((dT * (U - U.mean(axis=0))).sum() / var)
        s = min(AJUSTE_ESCALA[1] * escala_base, max(AJUSTE_ESCALA[0] * escala_base, s))
        return s, U.mean(axis=0) - s * T.mean(axis=0)

    def _proyectar(self, simbolos, letra: str, cursor, segmentos_usuario=None, escala_base: float = 1.0) -> list:
        """Tramos con lo que le falta a la letra, desde el cursor.

        Si se puede, la plantilla se ajusta a lo dibujado (`_ajustar`): lo que falta cae en
        la posición y el tamaño que corresponden a la letra del usuario (el final de una
        «n» llega a SU línea base). Para no saltar, el primer tramo arranca en el cursor y
        se acerca a la posición ajustada durante TRANSICION del alto de la letra. Sin
        ajuste, lo que falta se pega al cursor con la escala estimada."""
        plantilla = self.plantillas.get(letra)
        if plantilla is None:
            # Sin plantilla: proyección con los símbolos más probables del modelo.
            _, restantes = self.motor.project(np.asarray(simbolos, dtype=int), letra=letra)
            return [self.extractor.symbols_to_points(restantes, start=cursor)] if restantes else []

        alineacion = self._alinear(simbolos, letra, plantilla)
        if alineacion is None:
            return []
        tau, escala = alineacion
        restantes = plantilla.segmentos[tau + 1:]
        c = np.array(self._plano(cursor))
        ajuste = None
        if segmentos_usuario is not None:
            ajuste = self._ajustar(simbolos, segmentos_usuario, letra, plantilla, tau, escala_base)
        if ajuste is None:
            s, t = escala * escala_base, c - escala * escala_base * np.array(self._plano(plantilla.segmentos[tau][1]))
            transicion = 0.0
        else:
            s, t = ajuste
            ys = [self._plano(b)[1] for _, b, salto in plantilla.segmentos if not salto]
            transicion = TRANSICION * s * (max(ys) - min(ys))

        def llevar(p):
            return s * np.array(self._plano(p)) + t

        previo = llevar(plantilla.segmentos[tau][1])
        delta = c - previo                                     # cuánto hay que correrse para salir del cursor
        recorrido, primer_tramo = 0.0, True
        tramos = [[c]]
        for a, b, salto in restantes:
            q = llevar(b)
            if salto:
                primer_tramo = False
                tramos.append([q])                             # lápiz arriba: el siguiente tramo empieza en otro lado
                continue
            if primer_tramo:
                recorrido += float(np.linalg.norm(q - previo))
                previo = q
                w = 1.0 if transicion <= 0 else max(0.0, 1.0 - recorrido / transicion)
                q = q + w * delta
            tramos[-1].append(q)
        salida = []
        for k, tramo in enumerate(tramos):
            puntos = [p for p in (self._desde_plano(q) for q in tramo) if p is not None]
            # Un tramo de un solo punto después de levantar el lápiz es un punto (el de la
            # «i»): se dibuja como un punto en vez de descartarlo.
            if len(puntos) > 1:
                salida.append(puntos)
            elif len(puntos) == 1 and k > 0:
                salida.append(puntos * 2)
        return salida

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
            # Esperan al lápiz arriba los trazos terminados muy cortos (un punto) y los que
            # quedan enteros POR ENCIMA del trazo en curso (punto, virgulilla). El largo solo
            # no basta: en el dataset de un diseño por letra la virgulilla mide ~0.26, tanto
            # como un trazo principal corto. El trazo en curso siempre cuenta.
            terminados = [t for t in trazos[:-1] if not es_secundario_previo(t, trazos[-1])]
            # Los terminados van en orden canónico y el trazo en curso al final, que es
            # donde está el cursor y desde donde se proyecta lo que falta.
            trazos = self.extractor.ordenar_trazos(terminados) + [trazos[-1]]

        simbolos = self.extractor.extract_trazos(trazos)
        if len(simbolos) < MIN_SIMBOLOS:
            return Resultado("pocos", simbolos=simbolos)

        r = self._evaluar(simbolos, dibujando)
        proyeccion = []
        if r.estado != "no_definido":
            proyeccion = self._proyectar(r.simbolos, r.letra, cursor, self._segmentos(trazos)[1])
        return Resultado(r.estado, r.simbolos, r.probabilidades, r.letra, proyeccion, r.verosimilitud)

    def _evaluar(self, simbolos, dibujando: bool) -> Resultado:
        """Clasificación y «no definido» de una secuencia (sin proyección). La comparten
        los modos 2D y 3D: solo cambia de dónde salen los símbolos."""
        clave = (tuple(simbolos), dibujando)
        if clave != self._cache_clave:
            # Solo se recalcula el HMM cuando aparece un símbolo nuevo o se levanta el lápiz.
            letra, probs = self.motor.classify(np.asarray(simbolos, dtype=int), temperatura=TEMPERATURA,
                                               completa=not dibujando, peso_final=PESO_FINAL,
                                               largos=self.largos, peso_largo=PESO_LARGO)
            ordenadas = sorted(((l, float(p)) for l, p in probs.items()), key=lambda x: -x[1])
            vps = self.verosimilitud_por_simbolo(simbolos, letra)
            # «No definido» solo se decide con el lápiz arriba: a mitad de un trazo,
            # una letra real poco común puede pasar un momento por debajo del umbral.
            no_definido = (not dibujando) and len(simbolos) >= MIN_SIMBOLOS_JUICIO and vps < self.umbral
            self._cache_clave = clave
            self._cache = Resultado("no_definido" if no_definido else "reconocido",
                                    simbolos=simbolos, probabilidades=ordenadas, letra=letra,
                                    verosimilitud=vps)
        return self._cache
