"""
FeatureExtractor.py  (primera entrega: solo 2D)
===============================================
Módulo de procesamiento geométrico del proyecto Glifo.

Convierte un trazo continuo (lista de puntos del ratón) en una secuencia de
SÍMBOLOS DISCRETOS (enteros 0..K-1), que es lo que consume el HMM, y hace el
camino inverso (símbolos -> puntos) para dibujar la proyección punteada.

Solo usa la librería estándar (`math`).

CONVENCIONES (ver docs/contrato_features.md)
--------------------------------------------
* Símbolo s ∈ {0, ..., K-1} = sector angular centrado en  s · 360°/K.
  Con K = 8:  0=→  1=↗  2=↑  3=↖  4=←  5=↙  6=↓  7=↘
* Puntos (x, y) NORMALIZADOS: x/L, y/L, con L = lado del lienzo. El eje Y es el
  de pantalla (crece hacia abajo); este módulo lo invierte internamente para que
  "arriba" sea el símbolo 2.
* El paso de remuestreo `step` está en las mismas unidades que los puntos
  (fracción del lienzo).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

TWO_PI = 2.0 * math.pi

# Las 27 letras minúsculas del alfabeto español (sin dígrafos ni tildes).
ALPHABET = list("abcdefghijklmnñopqrstuvwxyz")

VALID_K = (8, 16)
VALID_MODES = ("2d", "3d")

# Valores por defecto: fijados con los datos reales (ver docs/contrato_features.md).
DEFAULT_K = 8
DEFAULT_STEP = 0.03
DEFAULT_SMOOTH_PASSES = 1


# ---------------------------------------------------------------------------
# Cuantización angular (Chain Code)
# ---------------------------------------------------------------------------
def quantize_angle(angle: float, k: int) -> int:
    """Ángulo en radianes -> sector 0..k-1.

    Se suma medio sector antes de dividir para que las direcciones "puras"
    (0°, 90°, 180°...) caigan en el CENTRO de su sector y no en la frontera.
    """
    sector = TWO_PI / k
    return int(((angle + sector / 2.0) % TWO_PI) // sector) % k


def symbol_to_angle(symbol: int, k: int) -> float:
    """Sector -> ángulo central del sector (radianes)."""
    return symbol * TWO_PI / k


# ---------------------------------------------------------------------------
# Validación de entradas
# ---------------------------------------------------------------------------
def _check_mode(mode: str) -> str:
    """Acepta solo '2d' o '3d'. Un valor desconocido NO cae en silencio a 2D."""
    if mode not in VALID_MODES:
        raise ValueError(f"mode debe ser '2d' o '3d', no {mode!r}.")
    if mode == "3d":
        raise NotImplementedError("El modo 3D llega en la segunda entrega.")
    return mode


def _check_point(p: Sequence[float]) -> tuple:
    """Punto (x, y) como tupla de floats; rechaza dimensión errónea, NaN e infinitos."""
    try:
        q = tuple(float(c) for c in p)
    except (TypeError, ValueError):
        raise ValueError(f"Punto no numérico: {p!r}.") from None
    if len(q) != 2:
        raise ValueError(f"Se esperaba un punto (x, y) y llegó {q!r}.")
    if not (math.isfinite(q[0]) and math.isfinite(q[1])):
        raise ValueError(f"Punto con valores no finitos (NaN/inf): {q!r}.")
    return q


# ---------------------------------------------------------------------------
# Preprocesamiento geométrico
# ---------------------------------------------------------------------------
def _dedupe(points: Sequence[Sequence[float]], eps: float = 1e-9) -> list:
    """Valida cada punto y elimina los consecutivos repetidos (el ratón repite eventos)."""
    out: list = []
    for p in points:
        q = _check_point(p)
        if not out or math.dist(out[-1], q) > eps:
            out.append(q)
    return out


def _smooth(points: list, passes: int) -> list:
    """Media móvil (1, 2, 1)/4 que quita el temblor, dejando fijos los extremos."""
    pts = list(points)
    for _ in range(passes):
        if len(pts) < 3:
            break
        nuevo = [pts[0]]
        for i in range(1, len(pts) - 1):
            nuevo.append(
                (
                    (pts[i - 1][0] + 2.0 * pts[i][0] + pts[i + 1][0]) / 4.0,
                    (pts[i - 1][1] + 2.0 * pts[i][1] + pts[i + 1][1]) / 4.0,
                )
            )
        nuevo.append(pts[-1])
        pts = nuevo
    return pts


def _resample(points: list, step: float) -> list:
    """Remuestreo por LONGITUD DE ARCO: un punto cada `step` recorrido.

    Así la velocidad del ratón no cambia la secuencia: dibujar lento o rápido
    produce los mismos símbolos. El tramo final incompleto (< step) no genera
    punto, lo que estabiliza la predicción en tiempo real.
    """
    if not points:
        return []
    out = [points[0]]
    cur = points[0]
    faltante = step
    for sig in points[1:]:
        seg = math.dist(cur, sig)
        while seg >= faltante - 1e-9 and seg > 1e-12:
            t = min(1.0, faltante / seg)
            cur = (cur[0] + (sig[0] - cur[0]) * t, cur[1] + (sig[1] - cur[1]) * t)
            out.append(cur)
            faltante = step
            seg = math.dist(cur, sig)
        faltante -= seg
        cur = sig
    return out


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------
class FeatureExtractor:
    """Trazo continuo <-> secuencia de símbolos discretos (2D).

    Parámetros
    ----------
    k             : número de sectores angulares: 8 o 16. Es el tamaño del alfabeto V.
    step          : paso de remuestreo, en fracción del lado del lienzo.
    smooth_passes : pasadas de suavizado previo (0 = sin suavizar).
    """

    def __init__(
        self,
        k: int = DEFAULT_K,
        step: float = DEFAULT_STEP,
        smooth_passes: int = DEFAULT_SMOOTH_PASSES,
    ):
        if k not in VALID_K:
            raise ValueError(f"k debe ser 8 o 16 (el enunciado lo fija), no {k!r}.")
        if not math.isfinite(step) or step <= 1e-6:
            raise ValueError("step debe ser un número finito y positivo.")
        if not isinstance(smooth_passes, int) or smooth_passes < 0:
            raise ValueError("smooth_passes debe ser un entero >= 0.")
        self.k = k
        self.step = float(step)
        self.smooth_passes = smooth_passes

    # ------------------------------------------------------------- utilidades
    @staticmethod
    def join_strokes(strokes: Sequence[Sequence[Sequence[float]]]) -> list:
        """Une varios trazos (levantar el lápiz) en una sola polilínea.

        El salto entre el final de un trazo y el inicio del siguiente queda como
        un segmento recto más: el HMM lo aprende como parte de la letra.
        """
        unidos: list = []
        for s in strokes:
            unidos.extend(tuple(p) for p in s)
        return unidos

    @staticmethod
    def ordenar_trazos(strokes: Sequence[Sequence[Sequence[float]]]) -> list:
        """Orden canónico de los trazos de una letra, para que no importe en qué
        orden se dibujaron.

        * Trazos principales (largo >= 50 % del más largo): de izquierda a
          derecha según su punto de inicio.
        * Trazos secundarios (punto de la «i», virgulilla de la «ñ», cruz de la
          «t»...): al final, de arriba abajo y de izquierda a derecha.

        Entrenamiento y GUI deben usar el mismo orden: por eso vive aquí.
        """
        trazos = [[tuple(p) for p in s] for s in strokes if len(s) > 0]
        if len(trazos) < 2:
            return trazos

        def largo(t):
            return sum(math.dist(a, b) for a, b in zip(t, t[1:]))

        largos = [largo(t) for t in trazos]
        tope = max(largos)
        principales = [t for t, l in zip(trazos, largos) if l >= 0.5 * tope]
        secundarios = [t for t, l in zip(trazos, largos) if l < 0.5 * tope]
        principales.sort(key=lambda t: t[0][0])
        secundarios.sort(key=lambda t: (t[0][1], t[0][0]))
        return principales + secundarios

    def extract_trazos(self, strokes: Sequence[Sequence[Sequence[float]]], mode: str = "2d",
                       con_saltos: bool = False):
        """Varios trazos -> una sola secuencia de símbolos.

        Los trazos se ordenan con `ordenar_trazos` y entre uno y otro se agrega
        UN símbolo con la dirección del salto del lápiz (del final de un trazo
        al inicio del siguiente). Así el salto conserva la posición relativa
        (el punto va arriba del palito) sin pesar tanto como un trazo largo.

        Con `con_saltos=True` devuelve (símbolos, índices de los símbolos de salto).
        """
        _check_mode(mode)
        trazos = self.ordenar_trazos(strokes)
        simbolos: list = []
        saltos: list = []
        for i, t in enumerate(trazos):
            if i > 0:
                (x0, y0), (x1, y1) = _check_point(trazos[i - 1][-1]), _check_point(t[0])
                if math.hypot(x1 - x0, y1 - y0) > 1e-9:
                    saltos.append(len(simbolos))
                    simbolos.append(quantize_angle(math.atan2(-(y1 - y0), x1 - x0), self.k))
            simbolos.extend(self.extract(t, mode))
        return (simbolos, saltos) if con_saltos else simbolos

    # ---------------------------------------------------------- preprocesado
    def preprocess(self, points: Sequence[Sequence[float]], mode: str = "2d") -> list:
        """Puntos crudos -> puntos limpios y equiespaciados (para depurar/dibujar)."""
        _check_mode(mode)
        pts = _dedupe(points)
        pts = _smooth(pts, self.smooth_passes)
        return _resample(pts, self.step)

    # -------------------------------------------------------------- extracción
    def extract(self, points: Sequence[Sequence[float]], mode: str = "2d") -> list:
        """Trazo 2D -> lista de símbolos (n puntos remuestreados -> n-1 símbolos)."""
        pts = self.preprocess(points, mode)
        simbolos = []
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            # El eje Y de pantalla crece hacia abajo: se invierte para que 90° = arriba.
            angulo = math.atan2(-(y1 - y0), x1 - x0)
            simbolos.append(quantize_angle(angulo, self.k))
        return simbolos

    def extract_2d(self, points: Sequence[Sequence[float]]) -> list:
        """Alias explícito de `extract(points, "2d")`."""
        return self.extract(points, "2d")

    # ----------------------------------------------------- reconstrucción inversa
    def symbols_to_points(
        self,
        symbols: Sequence[int],
        start: Sequence[float],
        mode: str = "2d",
        step: Optional[float] = None,
    ) -> list:
        """Símbolos -> coordenadas partiendo de `start` (misma convención que extract).

        Es lo que usa la GUI para la proyección punteada: los símbolos restantes
        {O_t+1..O_T} que entrega el HMM y la posición actual del cursor.
        """
        _check_mode(mode)
        paso = self.step if step is None else step
        if not math.isfinite(paso) or paso <= 0:
            raise ValueError("step debe ser un número finito y positivo.")
        x, y = _check_point(start)
        pts = [(x, y)]
        for s in symbols:
            if isinstance(s, bool) or int(s) != s or not 0 <= int(s) < self.k:
                raise ValueError(f"Símbolo fuera de rango 0..{self.k - 1}: {s!r}.")
            a = symbol_to_angle(int(s), self.k)
            x += paso * math.cos(a)
            y -= paso * math.sin(a)  # hacia arriba = Y de pantalla decreciente
            pts.append((x, y))
        return pts
