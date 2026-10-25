"""
FeatureExtractor.py
===================
Módulo de procesamiento geométrico del proyecto Glifo.

Convierte un trazo continuo (lista de puntos del ratón) en una secuencia de
SÍMBOLOS DISCRETOS (enteros 0..K-1) que es lo que consume el HMM, y hace el
camino inverso (símbolos -> puntos) para dibujar la proyección punteada.

Solo usa la librería estándar (`math`): no hay dependencias.

CONVENCIONES (el equipo debe respetarlas; ver docs/contrato_features.md)
------------------------------------------------------------------------
* Símbolo s ∈ {0, ..., K-1} = sector angular centrado en  s · 2π/K.
  Con K = 8:  0=→  1=↗  2=↑  3=↖  4=←  5=↙  6=↓  7=↘   (↑ es "arriba" en pantalla).
* 2D: puntos (x, y) normalizados al lienzo (x/L, y/L, con L = lado del lienzo).
      El eje Y es el de pantalla (crece hacia abajo); este módulo lo invierte
      internamente, de modo que 90° = arriba.
* 3D: puntos (x, y, z) sobre la esfera UNITARIA, en el marco de la CÁMARA:
      x → derecha, y → arriba, z → hacia el observador.
      Colatitud θ medida desde +Y y longitud φ = atan2(x, z); así el centro de
      la parte visible de la esfera cae sobre el ecuador (no sobre un polo
      singular) y "derecha/arriba" coinciden con las direcciones 2D.
* Paso de remuestreo: en 2D es una fracción del lienzo; en 3D es un ángulo en
  radianes (longitud de arco sobre la esfera unitaria).
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

TWO_PI = 2.0 * math.pi

# Las 27 letras minúsculas del alfabeto español (sin dígrafos ni tildes).
ALPHABET = list("abcdefghijklmnñopqrstuvwxyz")

Point2 = tuple
Point3 = tuple


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
# Utilidades geométricas internas
# ---------------------------------------------------------------------------
def _dedupe(points: Sequence[Sequence[float]], eps: float = 1e-9) -> list:
    """Elimina puntos consecutivos repetidos (el ratón suele repetir eventos)."""
    out: list = []
    for p in points:
        q = tuple(float(c) for c in p)
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
                tuple((a + 2.0 * b + c) / 4.0 for a, b, c in zip(pts[i - 1], pts[i], pts[i + 1]))
            )
        nuevo.append(pts[-1])
        pts = nuevo
    return pts


def _resample(points: list, step: float, dist: Callable, lerp: Callable) -> list:
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
        seg = dist(cur, sig)
        while seg >= faltante - 1e-9 and seg > 1e-12:
            t = min(1.0, faltante / seg)
            cur = lerp(cur, sig, t)
            out.append(cur)
            faltante = step
            seg = dist(cur, sig)
        faltante -= seg
        cur = sig
    return out


def _lerp2(a: Point2, b: Point2, t: float) -> Point2:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _norm3(v: Sequence[float]) -> Point3:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n == 0.0:
        raise ValueError("No se puede normalizar el vector nulo (punto fuera de la esfera).")
    return (v[0] / n, v[1] / n, v[2] / n)


def _dot3(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross3(a: Sequence[float], b: Sequence[float]) -> Point3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _arc(a: Point3, b: Point3) -> float:
    """Distancia angular (arco de círculo máximo) entre dos puntos de la esfera.

    atan2(|a×b|, a·b) es mucho más estable que acos(a·b) para ángulos pequeños.
    """
    c = _cross3(a, b)
    return math.atan2(math.sqrt(_dot3(c, c)), _dot3(a, b))


def _slerp(a: Point3, b: Point3, t: float) -> Point3:
    """Interpolación esférica: punto a fracción t del arco a -> b."""
    omega = _arc(a, b)
    if omega < 1e-9:
        return b
    s = math.sin(omega)
    wa = math.sin((1.0 - t) * omega) / s
    wb = math.sin(t * omega) / s
    return _norm3((wa * a[0] + wb * b[0], wa * a[1] + wb * b[1], wa * a[2] + wb * b[2]))


def _wrap_pi(a: float) -> float:
    """Lleva un ángulo a (-π, π]. Resuelve el salto de φ en ±π."""
    return (a + math.pi) % TWO_PI - math.pi


def _tangent_basis(p: Point3) -> tuple:
    """Base (norte, este) del plano tangente a la esfera en p.

    norte = hacia el polo +Y;  este = norte × p  (sentido en que crece φ).
    En el polo la dirección norte no está definida: se fija una por convención.
    """
    y_axis = (0.0, 1.0, 0.0)
    d = _dot3(y_axis, p)
    n = (y_axis[0] - d * p[0], y_axis[1] - d * p[1], y_axis[2] - d * p[2])
    largo = math.sqrt(_dot3(n, n))
    if largo < 1e-9:  # estamos en un polo
        este = (1.0, 0.0, 0.0)
        norte = _cross3(p, este)
        return norte, este
    norte = (n[0] / largo, n[1] / largo, n[2] / largo)
    este = _cross3(norte, p)
    return norte, este


def to_spherical(x: float, y: float, z: float) -> tuple:
    """(x, y, z) -> (θ, φ): colatitud desde +Y en [0, π] y longitud en (-π, π]."""
    r = math.sqrt(x * x + y * y + z * z)
    if r == 0.0:
        raise ValueError("El punto (0, 0, 0) no está sobre la esfera.")
    theta = math.acos(max(-1.0, min(1.0, y / r)))
    phi = math.atan2(x, z)
    return theta, phi


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------
class FeatureExtractor:
    """Trazo continuo <-> secuencia de símbolos discretos.

    Parámetros
    ----------
    k        : número de sectores angulares (8 o 16). Es el tamaño del alfabeto V.
    step_2d  : paso de remuestreo en 2D (fracción del lado del lienzo).
    step_3d  : paso de remuestreo en 3D (radianes sobre la esfera unitaria).
    smooth_passes : pasadas de suavizado previo (0 = sin suavizar).
    """

    def __init__(
        self,
        k: int = 8,
        step_2d: float = 0.04,
        step_3d: float = 0.08,
        smooth_passes: int = 1,
    ):
        if k < 4:
            raise ValueError("k debe ser >= 4 (se recomiendan 8 o 16).")
        if step_2d <= 1e-6 or step_3d <= 1e-6:
            raise ValueError("Los pasos de remuestreo deben ser positivos.")
        self.k = k
        self.step_2d = step_2d
        self.step_3d = step_3d
        self.smooth_passes = smooth_passes

    # ------------------------------------------------------------- utilidades
    @staticmethod
    def join_strokes(strokes: Sequence[Sequence[Sequence[float]]]) -> list:
        """Une varios trazos (levantar el lápiz) en una sola polilínea.

        El salto entre el final de un trazo y el inicio del siguiente queda
        como un segmento recto más: el HMM lo aprende como parte de la letra.
        """
        unidos: list = []
        for s in strokes:
            unidos.extend(tuple(p) for p in s)
        return unidos

    @staticmethod
    def to_spherical(x: float, y: float, z: float) -> tuple:
        return to_spherical(x, y, z)

    # ---------------------------------------------------------- preprocesado
    def preprocess_2d(self, points: Sequence[Point2]) -> list:
        """Puntos crudos -> puntos limpios y equiespaciados (2D)."""
        pts = _dedupe(points)
        pts = _smooth(pts, self.smooth_passes)
        return _resample(pts, self.step_2d, math.dist, _lerp2)

    def preprocess_3d(self, points: Sequence[Point3]) -> list:
        """Puntos crudos -> puntos limpios y equiespaciados sobre la esfera."""
        pts = _dedupe(points)
        pts = [_norm3(p) for p in pts]  # proyecta a la esfera unitaria
        pts = [_norm3(p) for p in _smooth(pts, self.smooth_passes)]
        return _resample(pts, self.step_3d, _arc, _slerp)

    def preprocess(self, points: Sequence[Sequence[float]], mode: str = "2d") -> list:
        return self.preprocess_3d(points) if mode == "3d" else self.preprocess_2d(points)

    # -------------------------------------------------------------- extracción
    def extract_2d(self, points: Sequence[Point2]) -> list:
        """Trazo 2D -> lista de símbolos (n puntos remuestreados -> n-1 símbolos)."""
        pts = self.preprocess_2d(points)
        simbolos = []
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            # El eje Y de pantalla crece hacia abajo: se invierte para que 90° = arriba.
            angulo = math.atan2(-(y1 - y0), x1 - x0)
            simbolos.append(quantize_angle(angulo, self.k))
        return simbolos

    def extract_3d(self, points: Sequence[Point3]) -> list:
        """Trazo sobre la esfera -> lista de símbolos a partir de (Δθ, Δφ).

        Δφ se envuelve a (-π, π] (el salto 179° -> -179° vale 2°, no -358°) y se
        escala por sen(θ), porque los meridianos convergen hacia los polos.
        """
        pts = self.preprocess_3d(points)
        esf = [to_spherical(*p) for p in pts]
        simbolos = []
        for (t0, f0), (t1, f1) in zip(esf, esf[1:]):
            d_theta = t1 - t0
            d_phi = _wrap_pi(f1 - f0)
            este = math.sin((t0 + t1) / 2.0) * d_phi
            norte = -d_theta  # θ disminuye al subir
            simbolos.append(quantize_angle(math.atan2(norte, este), self.k))
        return simbolos

    def extract(self, points: Sequence[Sequence[float]], mode: str = "2d") -> list:
        """Atajo: elige 2D o 3D según `mode` ('2d' | '3d')."""
        return self.extract_3d(points) if mode == "3d" else self.extract_2d(points)

    # ----------------------------------------------------- reconstrucción inversa
    def symbols_to_points_2d(
        self, symbols: Sequence[int], start: Point2, step: Optional[float] = None
    ) -> list:
        """Símbolos -> coordenadas 2D partiendo de `start` (misma convención que extract_2d)."""
        paso = self.step_2d if step is None else step
        x, y = float(start[0]), float(start[1])
        pts = [(x, y)]
        for s in symbols:
            a = symbol_to_angle(int(s), self.k)
            x += paso * math.cos(a)
            y -= paso * math.sin(a)  # hacia arriba = Y de pantalla decreciente
            pts.append((x, y))
        return pts

    def symbols_to_points_3d(
        self, symbols: Sequence[int], start: Point3, step: Optional[float] = None
    ) -> list:
        """Símbolos -> puntos sobre la esfera, avanzando por geodésicas.

        En cada paso se toma la base (norte, este) local, se elige la dirección
        del símbolo y se avanza `step` radianes por el círculo máximo.
        """
        paso = self.step_3d if step is None else step
        p = _norm3(start)
        pts = [p]
        for s in symbols:
            a = symbol_to_angle(int(s), self.k)
            norte, este = _tangent_basis(p)
            d = (
                math.cos(a) * este[0] + math.sin(a) * norte[0],
                math.cos(a) * este[1] + math.sin(a) * norte[1],
                math.cos(a) * este[2] + math.sin(a) * norte[2],
            )
            c, s_ = math.cos(paso), math.sin(paso)
            p = _norm3((c * p[0] + s_ * d[0], c * p[1] + s_ * d[1], c * p[2] + s_ * d[2]))
            pts.append(p)
        return pts

    def symbols_to_points(
        self,
        symbols: Sequence[int],
        start: Sequence[float],
        mode: str = "2d",
        step: Optional[float] = None,
    ) -> list:
        """Atajo para la proyección punteada: símbolos restantes + posición del cursor."""
        if mode == "3d":
            return self.symbols_to_points_3d(symbols, start, step)
        return self.symbols_to_points_2d(symbols, start, step)
