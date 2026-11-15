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
# Geometría sobre la esfera (modo 3D)
# ---------------------------------------------------------------------------
# Ángulo (radianes) que equivale al lado del lienzo 2D cuando la esfera está sin zoom.
# El paso 3D por defecto es DEFAULT_STEP de ese ángulo, para que una letra de tamaño
# cómodo dé más o menos tantos símbolos como en el plano.
ANGULO_LIENZO = 1.6
DEFAULT_PASO_ANGULAR = DEFAULT_STEP * ANGULO_LIENZO
Z_VISIBLE = 0.0   # z > Z_VISIBLE: el punto está en la cara que se ve


def _check_point3(p: Sequence[float]) -> tuple:
    """Punto (x, y, z) llevado a la esfera unitaria; rechaza NaN, infinitos y el origen."""
    try:
        q = tuple(float(c) for c in p)
    except (TypeError, ValueError):
        raise ValueError(f"Punto no numérico: {p!r}.") from None
    if len(q) != 3:
        raise ValueError(f"Se esperaba un punto (x, y, z) y llegó {q!r}.")
    if not all(math.isfinite(c) for c in q):
        raise ValueError(f"Punto con valores no finitos (NaN/inf): {q!r}.")
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2])
    if n < 1e-12:
        raise ValueError("El punto (0, 0, 0) no está sobre la esfera.")
    return (q[0] / n, q[1] / n, q[2] / n)


def a_esfericas(p: Sequence[float]) -> tuple:
    """(x, y, z) en el marco de la vista -> (θ, φ). θ: acimut, φ: ángulo polar desde arriba."""
    x, y, z = _check_point3(p)
    return math.atan2(x, z), math.acos(max(-1.0, min(1.0, y)))


def de_esfericas(theta: float, phi: float) -> tuple:
    """(θ, φ) -> (x, y, z). Inversa de `a_esfericas`."""
    s = math.sin(phi)
    return (s * math.sin(theta), math.cos(phi), s * math.cos(theta))


def angulo_entre(a: Sequence[float], b: Sequence[float]) -> float:
    """Distancia sobre la esfera (ángulo del círculo máximo) entre dos puntos unitarios."""
    cx = a[1] * b[2] - a[2] * b[1]
    cy = a[2] * b[0] - a[0] * b[2]
    cz = a[0] * b[1] - a[1] * b[0]
    return math.atan2(math.sqrt(cx * cx + cy * cy + cz * cz), a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def _slerp(a: Sequence[float], b: Sequence[float], t: float) -> tuple:
    """Punto a la fracción t del círculo máximo que va de a a b."""
    w = angulo_entre(a, b)
    if w < 1e-9:
        return tuple(a)
    sa, sb = math.sin((1 - t) * w) / math.sin(w), math.sin(t * w) / math.sin(w)
    return (sa * a[0] + sb * b[0], sa * a[1] + sb * b[1], sa * a[2] + sb * b[2])


def rumbo(a: Sequence[float], b: Sequence[float]) -> float:
    """Dirección del movimiento de a a b sobre la superficie, en radianes (0 = derecha,
    π/2 = arriba), calculada con las variaciones (Δθ, Δφ) de las coordenadas esféricas."""
    ta, pa = a_esfericas(a)
    tb, pb = a_esfericas(b)
    dtheta = (tb - ta + math.pi) % TWO_PI - math.pi      # el salto de -π a π no es un giro entero
    este = math.sin((pa + pb) / 2.0) * dtheta
    norte = -(pb - pa)
    return math.atan2(norte, este)


def base_local(p: Sequence[float]) -> tuple:
    """Vectores unitarios «este» y «norte» del plano tangente en p (marco de la vista)."""
    x, y, z = p
    n = math.hypot(x, z)
    if n < 1e-9:                     # justo en el polo: el este no está definido
        return (1.0, 0.0, 0.0), (0.0, 0.0, -1.0 if y > 0 else 1.0)
    e = (z / n, 0.0, -x / n)
    norte = (y * e[2] - z * e[1], z * e[0] - x * e[2], x * e[1] - y * e[0])   # p × e
    return e, norte


def mover(p: Sequence[float], direccion: float, distancia: float) -> tuple:
    """Avanza `distancia` (radianes) desde p por el círculo máximo con rumbo `direccion`."""
    e, n = base_local(p)
    c, s = math.cos(direccion), math.sin(direccion)
    t = (c * e[0] + s * n[0], c * e[1] + s * n[1], c * e[2] + s * n[2])
    cd, sd = math.cos(distancia), math.sin(distancia)
    q = (cd * p[0] + sd * t[0], cd * p[1] + sd * t[1], cd * p[2] + sd * t[2])
    m = math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2)
    return (q[0] / m, q[1] / m, q[2] / m)


def visibles(stroke: Sequence[Sequence[float]]) -> list:
    """Parte un trazo en los tramos que se ven (z > 0). Lo que queda detrás se descarta."""
    tramos, actual = [], []
    for p in stroke:
        q = _check_point3(p)
        if q[2] > Z_VISIBLE:
            actual.append(q)
        elif actual:
            tramos.append(actual)
            actual = []
    if actual:
        tramos.append(actual)
    return tramos


def _dedupe3(points: Sequence[Sequence[float]], eps: float = 1e-9) -> list:
    out: list = []
    for p in points:
        q = _check_point3(p)
        if not out or angulo_entre(out[-1], q) > eps:
            out.append(q)
    return out


def _smooth3(points: list, passes: int) -> list:
    """El mismo (1, 2, 1)/4 del plano, devolviendo cada punto a la esfera."""
    pts = list(points)
    for _ in range(passes):
        if len(pts) < 3:
            break
        nuevo = [pts[0]]
        for a, b, c in zip(pts, pts[1:], pts[2:]):
            v = tuple((a[i] + 2.0 * b[i] + c[i]) / 4.0 for i in range(3))
            m = math.sqrt(sum(x * x for x in v))
            nuevo.append(tuple(x / m for x in v))
        nuevo.append(pts[-1])
        pts = nuevo
    return pts


def _resample3(points: list, paso: float) -> list:
    """Remuestreo por longitud de arco sobre la esfera: un punto cada `paso` radianes."""
    if not points:
        return []
    out = [points[0]]
    cur = points[0]
    faltante = paso
    for sig in points[1:]:
        seg = angulo_entre(cur, sig)
        while seg >= faltante - 1e-12 and seg > 1e-12:
            cur = _slerp(cur, sig, min(1.0, faltante / seg))
            out.append(cur)
            faltante = paso
            seg = angulo_entre(cur, sig)
        faltante -= seg
        cur = sig
    return out


def _largo3(t: Sequence[Sequence[float]]) -> float:
    return sum(angulo_entre(a, b) for a, b in zip(t, t[1:]))


def pegar_en_esfera(trazos, theta0: float, phi0: float, lado: float, giro: float = 0.0) -> list:
    c = de_esfericas(theta0, phi0)
    e, n = base_local(c)
    cg, sg = math.cos(giro), math.sin(giro)
    salida = []
    for trazo in trazos:
        nuevo = []
        for x, y in trazo:
            u = (x - 0.5) * lado
            w = -(y - 0.5) * lado            # en pantalla la y crece hacia abajo
            u, w = cg * u - sg * w, sg * u + cg * w
            r = math.hypot(u, w)
            if r < 1e-12:
                nuevo.append(c)
                continue
            d = tuple((u * e[i] + w * n[i]) / r for i in range(3))
            nuevo.append(tuple(math.cos(r) * c[i] + math.sin(r) * d[i] for i in range(3)))
        salida.append(nuevo)
    return salida


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------
class FeatureExtractor:
    """Trazo continuo <-> secuencia de símbolos discretos (2D).

    Parámetros
    ----------
    k             : número de sectores angulares: 8 o 16. Es el tamaño del alfabeto V.
    step          : paso de remuestreo 2D, en fracción del lado del lienzo.
    smooth_passes : pasadas de suavizado previo (0 = sin suavizar).
    paso_angular  : paso de remuestreo 3D, en radianes sobre la esfera.
    """

    def __init__(
        self,
        k: int = DEFAULT_K,
        step: float = DEFAULT_STEP,
        smooth_passes: int = DEFAULT_SMOOTH_PASSES,
        paso_angular: float = DEFAULT_PASO_ANGULAR,
    ):
        if k not in VALID_K:
            raise ValueError(f"k debe ser 8 o 16 (el enunciado lo fija), no {k!r}.")
        if not math.isfinite(step) or step <= 1e-6:
            raise ValueError("step debe ser un número finito y positivo.")
        if not isinstance(smooth_passes, int) or smooth_passes < 0:
            raise ValueError("smooth_passes debe ser un entero >= 0.")
        if not math.isfinite(paso_angular) or not 1e-6 < paso_angular < math.pi:
            raise ValueError("paso_angular debe estar entre 0 y π radianes.")
        self.k = k
        self.step = float(step)
        self.smooth_passes = smooth_passes
        self.paso_angular = float(paso_angular)

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

    @staticmethod
    def ordenar_trazos_3d(strokes: Sequence[Sequence[Sequence[float]]]) -> list:
        """El mismo orden canónico de `ordenar_trazos`, sobre la esfera: el largo se mide
        en ángulo, «izquierda» es menor θ y «arriba» es menor φ."""
        trazos = [[_check_point3(p) for p in s] for s in strokes if len(s) > 0]
        if len(trazos) < 2:
            return trazos
        largos = [_largo3(t) for t in trazos]
        tope = max(largos)
        principales = [t for t, l in zip(trazos, largos) if l >= 0.5 * tope]
        secundarios = [t for t, l in zip(trazos, largos) if l < 0.5 * tope]
        principales.sort(key=lambda t: a_esfericas(t[0])[0])
        secundarios.sort(key=lambda t: (a_esfericas(t[0])[1], a_esfericas(t[0])[0]))
        return principales + secundarios

    def extract_trazos(self, strokes: Sequence[Sequence[Sequence[float]]], mode: str = "2d",
                       con_saltos: bool = False):
        if _check_mode(mode) == "3d":
            tramos = [v for s in strokes for v in visibles(s)]
            trazos = self.ordenar_trazos_3d(tramos)
            simbolos, saltos = [], []
            for i, t in enumerate(trazos):
                if i > 0 and angulo_entre(trazos[i - 1][-1], t[0]) > 1e-9:
                    saltos.append(len(simbolos))
                    simbolos.append(quantize_angle(rumbo(trazos[i - 1][-1], t[0]), self.k))
                simbolos.extend(self.extract(t, "3d"))
            return (simbolos, saltos) if con_saltos else simbolos
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
        if _check_mode(mode) == "3d":
            return _resample3(_smooth3(_dedupe3(points), self.smooth_passes), self.paso_angular)
        pts = _dedupe(points)
        pts = _smooth(pts, self.smooth_passes)
        return _resample(pts, self.step)

    # -------------------------------------------------------------- extracción
    def extract(self, points: Sequence[Sequence[float]], mode: str = "2d") -> list:
        """Trazo -> lista de símbolos (n puntos remuestreados -> n-1 símbolos)."""
        pts = self.preprocess(points, mode)
        if mode == "3d":
            return [quantize_angle(rumbo(a, b), self.k) for a, b in zip(pts, pts[1:])]
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
        tres_d = _check_mode(mode) == "3d"
        paso = (self.paso_angular if tres_d else self.step) if step is None else step
        if not math.isfinite(paso) or paso <= 0:
            raise ValueError("step debe ser un número finito y positivo.")
        if tres_d:
            p = _check_point3(start)
            pts = [p]
            for s in symbols:
                if isinstance(s, bool) or int(s) != s or not 0 <= int(s) < self.k:
                    raise ValueError(f"Símbolo fuera de rango 0..{self.k - 1}: {s!r}.")
                p = mover(p, symbol_to_angle(int(s), self.k), paso)
                pts.append(p)
            return pts
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
