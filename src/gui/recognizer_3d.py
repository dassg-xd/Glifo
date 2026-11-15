from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from src.FeatureExtractor import (
    ANGULO_LIENZO,
    DEFAULT_PASO_ANGULAR,
    _largo3,
    a_esfericas,
    angulo_entre,
    base_local,
    centro_esferico,
    pegar_en_esfera,
)
from src.gui.recognizer import (
    CARPETA_DATOS,
    LARGO_SECUNDARIO,
    MIN_SIMBOLOS,
    RAIZ,
    Reconocedor,
    Resultado,
)

RUTA_MODELOS_3D = RAIZ / "data" / "modelos_letras_3d.json"
RUTA_SECUENCIAS_3D = RAIZ / "data" / "secuencias_3d_k16.json"
ZOOM_MIN, ZOOM_MAX = 0.6, 3.0


class Reconocedor3D(Reconocedor):
    """Reconocedor del modo 3D. Todo se mide en el marco de la ESFERA: cuenta la esfera
    completa (se vea o no) y «arriba» es su polo norte, así que el resultado no depende
    de cómo esté girada la vista."""

    def __init__(self, ruta_modelos: Path = RUTA_MODELOS_3D, ruta_secuencias: Path | None = RUTA_SECUENCIAS_3D,
                 carpeta_datos: Path | None = CARPETA_DATOS):
        super().__init__(ruta_modelos, ruta_secuencias, carpeta_datos)

    # ------------------------------------------------------- plantillas
    def _segmentos(self, trazos) -> tuple[list, list]:
        """Recibe la muestra 2D normalizada (así la llama `_construir_plantillas`), la pega
        sobre el ecuador, en (θ, φ) = (0, π/2), y devuelve sus símbolos esféricos y un
        segmento por símbolo."""
        fx = self.extractor
        fx.paso_angular = DEFAULT_PASO_ANGULAR
        en_esfera = pegar_en_esfera(trazos, 0.0, math.pi / 2, ANGULO_LIENZO)
        return fx.extract_trazos(en_esfera, "3d"), self._segmentos_3d(en_esfera)

    def _segmentos_3d(self, trazos) -> list:
        """Misma lógica que extract_trazos en modo 3D, guardando el segmento de cada símbolo."""
        fx = self.extractor
        ordenados = fx.ordenar_trazos_3d(trazos)
        segmentos = []
        for i, t in enumerate(ordenados):
            if i > 0 and angulo_entre(ordenados[i - 1][-1], t[0]) > 1e-9:
                segmentos.append((ordenados[i - 1][-1], t[0], True))
            puntos = fx.preprocess(t, "3d")
            segmentos.extend((puntos[j], puntos[j + 1], False) for j in range(len(puntos) - 1))
        return segmentos

    # ------------------------------------------------------- proyección
    # La proyección se ajusta en el «marco de la letra»: la esfera girada para que el centro
    # de lo dibujado quede en (0, 0, 1), con su este en +x y su norte en +y. Ahí la plantilla
    # (pegada en ese mismo punto) y el dibujo se comparan en el plano (x, y).
    @staticmethod
    def _marco_de_la_letra(trazos, cursor) -> np.ndarray:
        """Matriz 3×3 que lleva del marco de la esfera al marco de la letra."""
        c = centro_esferico(trazos) or tuple(cursor)
        e, n = base_local(c)
        return np.array([e, n, c], dtype=float)

    def _plano(self, p) -> tuple:
        return (float(p[0]), float(p[1]))

    def _desde_plano(self, q):
        """(x, y) del marco de la letra -> punto de la esfera (None si cae fuera del hemisferio)."""
        x, y = float(q[0]), float(q[1])
        d = x * x + y * y
        if d >= 1.0:
            return None
        return (x, y, math.sqrt(1.0 - d))

    def _proyectar_3d(self, simbolos, letra: str, trazos, cursor, zoom: float = 1.0) -> list:
        """Lo que le falta a la letra, desde el cursor, en el marco de la esfera."""
        Q = self._marco_de_la_letra(trazos, cursor)
        segmentos = [(tuple(Q @ np.asarray(a)), tuple(Q @ np.asarray(b)), salto)
                     for a, b, salto in self._segmentos_3d(trazos)]
        tramos = self._proyectar(simbolos, letra, tuple(Q @ np.asarray(cursor)), segmentos, escala_base=1.0 / zoom)
        return [[tuple(Q.T @ np.asarray(q)) for q in tramo] for tramo in tramos]

    # ------------------------------------------------------------- principal
    def procesar_3d(self, trazos, dibujando: bool = True, zoom: float = 1.0) -> Resultado:
        """`trazos`: lista de trazos; cada trazo, puntos (x, y, z) en el marco de la esfera.
        `zoom`: el de la cámara (1 = esfera completa a la vista); solo ajusta el tamaño del
        paso, para que una letra dibujada de cerca (más chica sobre la esfera) dé los mismos
        símbolos. La orientación de la vista no interviene."""
        trazos = [[tuple(float(c) for c in p) for p in t] for t in trazos if len(t)]
        if not trazos:
            self.reiniciar()
            return Resultado("vacio")
        zoom = min(ZOOM_MAX, max(ZOOM_MIN, float(zoom)))
        self.extractor.paso_angular = DEFAULT_PASO_ANGULAR / zoom

        cursor = trazos[-1][-1]
        if dibujando and len(trazos) > 1:
            # Igual que en 2D: mientras se dibuja, esperan al lápiz arriba los trazos
            # terminados muy cortos y los que quedan enteros POR ENCIMA del trazo en curso
            # (punto, virgulilla). «Encima» es más cerca del polo norte: menor φ.
            minimo = LARGO_SECUNDARIO * ANGULO_LIENZO / zoom
            phi_actual = min(a_esfericas(p)[1] for p in trazos[-1])

            def espera(t):
                return _largo3(t) < minimo or max(a_esfericas(p)[1] for p in t) < phi_actual
            trazos = [t for t in trazos[:-1] if not espera(t)] + [trazos[-1]]

        simbolos = self.extractor.extract_trazos(trazos, "3d")
        if len(simbolos) < MIN_SIMBOLOS:
            return Resultado("pocos", simbolos=simbolos)

        r = self._evaluar(simbolos, dibujando)
        proyeccion = [] if r.estado == "no_definido" else self._proyectar_3d(r.simbolos, r.letra, trazos, cursor, zoom)
        return Resultado(r.estado, r.simbolos, r.probabilidades, r.letra, proyeccion, r.verosimilitud)
