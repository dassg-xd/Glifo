from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from src.FeatureExtractor import (
    ANGULO_LIENZO,
    DEFAULT_PASO_ANGULAR,
    Z_VISIBLE,
    _largo3,
    a_esfericas,
    angulo_entre,
    pegar_en_esfera,
    visibles,
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
RUTA_SECUENCIAS_3D = RAIZ / "data" / "secuencias_3d_k8.json"
ZOOM_MIN, ZOOM_MAX = 0.6, 3.0


class Reconocedor3D(Reconocedor):
    def __init__(self, ruta_modelos: Path = RUTA_MODELOS_3D, ruta_secuencias: Path | None = RUTA_SECUENCIAS_3D,
                 carpeta_datos: Path | None = CARPETA_DATOS):
        super().__init__(ruta_modelos, ruta_secuencias, carpeta_datos)

    # ------------------------------------------------------- plantillas
    def _segmentos(self, trazos) -> tuple[list, list]:
        """Recibe la muestra 2D normalizada (así la llama `_construir_plantillas`), la pega
        en el centro de la vista y devuelve sus símbolos esféricos y un segmento por símbolo."""
        fx = self.extractor
        fx.paso_angular = DEFAULT_PASO_ANGULAR
        en_esfera = pegar_en_esfera(trazos, 0.0, math.pi / 2, ANGULO_LIENZO)
        return fx.extract_trazos(en_esfera, "3d"), self._segmentos_3d(en_esfera)

    def _segmentos_3d(self, trazos_vista) -> list:
        """Misma lógica que extract_trazos en modo 3D, guardando el segmento de cada símbolo."""
        fx = self.extractor
        ordenados = fx.ordenar_trazos_3d([v for t in trazos_vista for v in visibles(t)])
        segmentos = []
        for i, t in enumerate(ordenados):
            if i > 0 and angulo_entre(ordenados[i - 1][-1], t[0]) > 1e-9:
                segmentos.append((ordenados[i - 1][-1], t[0], True))
            puntos = fx.preprocess(t, "3d")
            segmentos.extend((puntos[j], puntos[j + 1], False) for j in range(len(puntos) - 1))
        return segmentos

    # ------------------------------------------------------- proyección
    def _plano(self, p) -> tuple:
        """La proyección se ajusta sobre lo que se ve: (x, y) del marco de la vista, que con
        la proyección ortográfica es exactamente la posición en pantalla."""
        return (float(p[0]), float(p[1]))

    def _desde_plano(self, q):
        """(x, y) de pantalla -> punto de la cara visible de la esfera (None si cae afuera)."""
        x, y = float(q[0]), float(q[1])
        d = x * x + y * y
        if d >= 1.0:
            return None
        return (x, y, math.sqrt(1.0 - d))

    # ------------------------------------------------------------- principal
    def procesar_3d(self, trazos, rotacion, dibujando: bool = True, zoom: float = 1.0) -> Resultado:
        """`trazos`: lista de trazos; cada trazo, puntos (x, y, z) en el marco de la esfera.
        `rotacion`: matriz 3×3 que lleva del marco de la esfera al de la vista.
        `zoom`: el de la cámara (1 = esfera completa a la vista)."""
        trazos = [t for t in trazos if len(t)]
        if not trazos:
            self.reiniciar()
            return Resultado("vacio")
        R = np.asarray(rotacion, dtype=float)
        zoom = min(ZOOM_MAX, max(ZOOM_MIN, float(zoom)))
        vista = [[tuple(R @ np.asarray(p, dtype=float)) for p in t] for t in trazos]
        self.extractor.paso_angular = DEFAULT_PASO_ANGULAR / zoom

        cursor = vista[-1][-1]
        if dibujando and len(vista) > 1:
            # Igual que en 2D: mientras se dibuja, esperan al lápiz arriba los trazos
            # terminados muy cortos y los que quedan enteros POR ENCIMA del trazo en curso
            # (punto, virgulilla). Todo se mide en lo que se ve; «encima» es menor φ.
            minimo = LARGO_SECUNDARIO * ANGULO_LIENZO / zoom
            actual = [p for v in visibles(vista[-1]) for p in v]

            def espera(t):
                tramos_t = visibles(t)
                if sum(_largo3(v) for v in tramos_t) < minimo:
                    return True
                if not actual:
                    return False
                phi_t = max(a_esfericas(p)[1] for v in tramos_t for p in v)
                return phi_t < min(a_esfericas(p)[1] for p in actual)
            vista = [t for t in vista[:-1] if not espera(t)] + [vista[-1]]

        simbolos = self.extractor.extract_trazos(vista, "3d")
        if len(simbolos) < MIN_SIMBOLOS:
            return Resultado("pocos", simbolos=simbolos)

        r = self._evaluar(simbolos, dibujando)
        proyeccion = []
        if r.estado != "no_definido" and cursor[2] > Z_VISIBLE:     # sin cursor a la vista no hay desde dónde proyectar
            Rt = R.T
            tramos = self._proyectar(r.simbolos, r.letra, cursor, self._segmentos_3d(vista), escala_base=1.0 / zoom)
            proyeccion = [[tuple(Rt @ np.asarray(p)) for p in tramo] for tramo in tramos]
        return Resultado(r.estado, r.simbolos, r.probabilidades, r.letra, proyeccion, r.verosimilitud)
