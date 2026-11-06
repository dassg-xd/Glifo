"""
recognizer.py
=============
Puente entre la interfaz y los módulos matemáticos.

Recibe los trazos del lienzo (puntos normalizados x/L, y/L) y devuelve lo que
la ventana 2D necesita mostrar:

1. FeatureExtractor: trazos -> símbolos O_1:t (chain code, K = 8).
2. HMMEngine.classify: P(c | O_1:t) para las 27 letras, con escalado de
   temperatura (T = 3) para que los porcentajes no exageren la confianza.
3. HMMEngine.project: símbolos restantes de la letra ganadora, que el
   FeatureExtractor convierte de nuevo en puntos desde el cursor.
4. «Trazo no definido»: si la verosimilitud por símbolo de la letra ganadora,
   log P(O_1:t | λ_ĉ) / t, queda por debajo del umbral, el trazo no se parece a
   ninguna letra. La probabilidad a posteriori no sirve para esto: al
   normalizar entre 27 modelos, hasta un garabato le da casi 100 % a alguna.
   El umbral es el percentil 7 de esa misma medida sobre las secuencias de
   entrenamiento (con los modelos de 8 estados detecta ~97 % de garabatos y
   marca ~1.5 % de letras reales). Si se reentrenan los modelos, conviene
   volver a revisar este percentil.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.FeatureExtractor import FeatureExtractor
from src.HMMEngine import HMMEngine

RAIZ = Path(__file__).resolve().parents[2]
RUTA_MODELOS = RAIZ / "data" / "modelos_letras_entrenados.json"
RUTA_SECUENCIAS = RAIZ / "data" / "secuencias_2d_k8.json"

MIN_SIMBOLOS = 4          # con menos, la clasificación no es fiable (contrato del extractor)
MIN_SIMBOLOS_JUICIO = 8   # no se declara «no definido» con trazos muy cortos
PERCENTIL_UMBRAL = 7
# Escalado de temperatura de P(c | O_1:t). Calibrado con el 20 % de prueba: con
# T = 3 la confianza media mostrada (63 %) se acerca al acierto real (68 %);
# con T = 1 el programa mostraba 81 % de confianza media y muchos 100 % errados.
TEMPERATURA = 3.0
UMBRAL_POR_DEFECTO = -1.55


@dataclass
class Resultado:
    estado: str                      # "vacio" | "pocos" | "reconocido" | "no_definido"
    simbolos: list = field(default_factory=list)
    probabilidades: list = field(default_factory=list)   # [(letra, p)] de mayor a menor
    letra: str | None = None
    proyeccion: list = field(default_factory=list)       # [(x, y)] normalizados, desde el cursor
    verosimilitud: float | None = None                   # log P / t de la letra ganadora


class Reconocedor:
    def __init__(self, ruta_modelos: Path = RUTA_MODELOS, ruta_secuencias: Path | None = RUTA_SECUENCIAS):
        self.motor = HMMEngine()
        self.motor.cargar_modelos(str(ruta_modelos))
        self.extractor = FeatureExtractor()
        self.umbral = self._calcular_umbral(ruta_secuencias)
        self._cache_simbolos: tuple | None = None
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
        self._cache_simbolos = None
        self._cache = None

    # ------------------------------------------------------------- principal
    def procesar(self, trazos) -> Resultado:
        """`trazos`: lista de trazos; cada trazo, lista de puntos (x, y) normalizados."""
        trazos = [t for t in trazos if t]
        if not trazos:
            self.reiniciar()
            return Resultado("vacio")

        simbolos = self.extractor.extract(self.extractor.join_strokes(trazos))
        if len(simbolos) < MIN_SIMBOLOS:
            return Resultado("pocos", simbolos=simbolos)

        cursor = tuple(trazos[-1][-1])
        clave = tuple(simbolos)
        if clave != self._cache_simbolos:
            # Solo se recalcula el HMM cuando aparece un símbolo nuevo.
            letra, probs = self.motor.classify(np.asarray(simbolos, dtype=int), temperatura=TEMPERATURA)
            ordenadas = sorted(((l, float(p)) for l, p in probs.items()), key=lambda x: -x[1])
            vps = self.verosimilitud_por_simbolo(simbolos, letra)
            no_definido = len(simbolos) >= MIN_SIMBOLOS_JUICIO and vps < self.umbral
            if no_definido:
                restantes = []
            else:
                _, restantes = self.motor.project(np.asarray(simbolos, dtype=int), letra=letra)
            self._cache_simbolos = clave
            self._cache = Resultado(
                "no_definido" if no_definido else "reconocido",
                simbolos=simbolos, probabilidades=ordenadas, letra=letra,
                proyeccion=restantes, verosimilitud=vps,
            )

        r = self._cache
        # La proyección arranca siempre en la posición actual del cursor.
        puntos = self.extractor.symbols_to_points(r.proyeccion, start=cursor) if r.proyeccion else []
        return Resultado(r.estado, r.simbolos, r.probabilidades, r.letra, puntos, r.verosimilitud)
