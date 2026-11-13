"""
canvas_3d.py
============
Lienzo 3D: una esfera que se puede girar y acercar, y sobre la que se dibuja.

* OpenGL (las clases de Qt, sin dependencias extra) dibuja la esfera iluminada
  con su malla de meridianos y paralelos, con proyección ORTOGRÁFICA: lo que se
  ve es exactamente la mitad con z > 0 en el marco de la vista, la misma regla
  que usa el análisis.
* Encima, con QPainter y el mismo estilo del lienzo 2D, van las líneas guía
  (fijas en pantalla: «arriba» siempre es arriba), los trazos, la proyección
  punteada y el cursor.

Controles: clic izquierdo dibuja, clic derecho (o el botón del medio) gira, la
rueda acerca o aleja.

Marcos: los trazos se guardan en el marco de la ESFERA (puntos unitarios), así
giran con ella. `rotacion` lleva del marco de la esfera al de la vista
(x a la derecha, y hacia arriba, z hacia quien mira).
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QMatrix4x4, QPainter, QPainterPath, QPen, QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QSizePolicy

from . import theme as T

ZOOM_MIN, ZOOM_MAX = 0.6, 3.0
RADIO_RELATIVO = 0.44          # radio de la esfera (zoom 1) como fracción del lado menor del lienzo
DISTANCIA_MINIMA = 0.002       # radianes: filtra eventos repetidos del ratón
SENSIBILIDAD_GIRO = 0.008      # radianes por píxel arrastrado
PASO_MALLA = 15                # grados entre meridianos y entre paralelos
ELEVACION = 1.002              # la malla se dibuja apenas por encima de la superficie

GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT = 0x4000, 0x0100
GL_DEPTH_TEST, GL_BLEND, GL_CULL_FACE, GL_MULTISAMPLE = 0x0B71, 0x0BE2, 0x0B44, 0x809D
GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA = 0x0302, 0x0303
GL_TRIANGLES, GL_LINES, GL_FLOAT, GL_LEQUAL = 0x0004, 0x0001, 0x1406, 0x0203

VERT = """#version 120
attribute vec3 posicion;
uniform mat4 mvp;
uniform mat4 modelo;
varying vec3 normal_vista;
void main() {
    normal_vista = mat3(modelo) * posicion;   // en una esfera unitaria la normal es la posicion
    gl_Position = mvp * vec4(posicion, 1.0);
}
"""
FRAG_ESFERA = """#version 120
varying vec3 normal_vista;
uniform vec3 color_base;
uniform vec3 color_borde;
void main() {
    vec3 n = normalize(normal_vista);
    vec3 luz = normalize(vec3(-0.45, 0.6, 0.85));
    float difusa = max(dot(n, luz), 0.0);
    float brillo = pow(max(dot(reflect(-luz, n), vec3(0.0, 0.0, 1.0)), 0.0), 40.0);
    float borde = pow(1.0 - max(n.z, 0.0), 3.0);            // el contorno se oscurece un poco
    vec3 c = color_base * (0.62 + 0.38 * difusa) + vec3(0.18 * brillo);
    gl_FragColor = vec4(mix(c, color_borde, 0.55 * borde), 1.0);
}
"""
FRAG_MALLA = """#version 120
varying vec3 normal_vista;
uniform vec4 color;
void main() {
    float frente = clamp(normal_vista.z * 4.0, 0.0, 1.0);   // la malla se desvanece hacia el borde
    gl_FragColor = vec4(color.rgb, color.a * (0.35 + 0.65 * frente));
}
"""


def _rot(eje: str, a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    if eje == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _malla_esfera(n_lat: int = 48, n_lon: int = 96) -> np.ndarray:
    """Triángulos de una esfera unitaria (lista plana de vértices, sin índices)."""
    lat = np.linspace(0, math.pi, n_lat + 1)
    lon = np.linspace(0, 2 * math.pi, n_lon + 1)

    def p(i, j):
        return (math.sin(lat[i]) * math.sin(lon[j]), math.cos(lat[i]), math.sin(lat[i]) * math.cos(lon[j]))
    v = []
    for i in range(n_lat):
        for j in range(n_lon):
            a, b, c, d = p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1)
            v += [a, b, c, a, c, d]
    return np.array(v, dtype=np.float32)


def _lineas_malla() -> np.ndarray:
    """Meridianos y paralelos cada PASO_MALLA grados, como pares de vértices (GL_LINES)."""
    v = []
    pasos = 120
    for lon in range(0, 360, PASO_MALLA):
        t = math.radians(lon)
        pts = [(math.sin(f) * math.sin(t), math.cos(f), math.sin(f) * math.cos(t))
               for f in np.linspace(0, math.pi, pasos)]
        v += [x for a, b in zip(pts, pts[1:]) for x in (a, b)]
    for lat in range(-90 + PASO_MALLA, 90, PASO_MALLA):
        f = math.radians(90 - lat)
        pts = [(math.sin(f) * math.sin(t), math.cos(f), math.sin(f) * math.cos(t))
               for t in np.linspace(0, 2 * math.pi, pasos)]
        v += [x for a, b in zip(pts, pts[1:]) for x in (a, b)]
    return (np.array(v, dtype=np.float32) * ELEVACION).astype(np.float32)


def _rgb(hexa: str) -> tuple:
    c = T.color(hexa)
    return (c.redF(), c.greenF(), c.blueF())


class Lienzo3D(QOpenGLWidget):
    trazos_cambiaron = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        formato = QSurfaceFormat()
        formato.setSamples(4)
        formato.setDepthBufferSize(24)
        self.setFormat(formato)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(360, 360)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.trazos: list[list[tuple]] = []           # marco de la esfera
        self.proyeccion: list[list[tuple]] = []       # marco de la esfera
        self.no_definido = False
        self.rotacion = np.eye(3)
        self.zoom = 1.0
        self._dibujando = False
        self._girando_desde: QPointF | None = None
        self._listo_gl = False

    # ------------------------------------------------------------ geometría
    @property
    def dibujando(self) -> bool:
        return self._dibujando

    @property
    def vacio(self) -> bool:
        return not any(self.trazos)

    def radio(self) -> float:
        return RADIO_RELATIVO * min(self.width(), self.height()) * self.zoom

    def centro(self) -> QPointF:
        return QPointF(self.width() / 2, self.height() / 2)

    def _a_vista(self, pos: QPointF):
        """Punto de pantalla -> punto de la esfera en el marco de la vista, o None si cae afuera."""
        r, c = self.radio(), self.centro()
        x, y = (pos.x() - c.x()) / r, (c.y() - pos.y()) / r
        d = x * x + y * y
        if d >= 1.0:
            return None
        return np.array([x, y, math.sqrt(1.0 - d)])

    def _pantalla(self, v) -> QPointF:
        r, c = self.radio(), self.centro()
        return QPointF(c.x() + v[0] * r, c.y() - v[1] * r)

    # ------------------------------------------------------------- acciones
    def limpiar(self) -> None:
        self.trazos = []
        self.proyeccion = []
        self.no_definido = False
        self._dibujando = False
        self.update()
        self.trazos_cambiaron.emit()

    def centrar_vista(self) -> None:
        self.rotacion = np.eye(3)
        self.zoom = 1.0
        self.update()
        self.trazos_cambiaron.emit()

    def mostrar_resultado(self, proyeccion, no_definido: bool) -> None:
        self.proyeccion = list(proyeccion)
        self.no_definido = no_definido
        self.update()

    # -------------------------------------------------------------- eventos
    def mousePressEvent(self, ev):
        if ev.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._girando_desde = ev.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        v = self._a_vista(ev.position())
        if v is None:
            return
        self._dibujando = True
        self.trazos.append([tuple(self.rotacion.T @ v)])
        self.update()
        self.trazos_cambiaron.emit()

    def mouseMoveEvent(self, ev):
        if self._girando_desde is not None:
            d = ev.position() - self._girando_desde
            self._girando_desde = ev.position()
            # Arrastrar a la derecha gira alrededor del eje vertical de la vista; hacia abajo,
            # alrededor del horizontal. Se aplica en el marco de la vista (multiplica por la izquierda).
            giro = _rot("y", d.x() * SENSIBILIDAD_GIRO) @ _rot("x", d.y() * SENSIBILIDAD_GIRO)
            self.rotacion = giro @ self.rotacion
            self.update()
            self.trazos_cambiaron.emit()        # lo que se ve cambió: se vuelve a analizar
            return
        if not self._dibujando:
            return
        v = self._a_vista(ev.position())
        if v is None:
            return
        p = tuple(self.rotacion.T @ v)
        u = self.trazos[-1][-1]
        if math.acos(max(-1.0, min(1.0, float(np.dot(p, u))))) >= DISTANCIA_MINIMA:
            self.trazos[-1].append(p)
            self.update()
            self.trazos_cambiaron.emit()

    def mouseReleaseEvent(self, ev):
        if ev.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._girando_desde = None
            self.setCursor(Qt.CursorShape.CrossCursor)
            return
        if ev.button() == Qt.MouseButton.LeftButton and self._dibujando:
            self._dibujando = False
            self.update()
            self.trazos_cambiaron.emit()   # al levantar el lápiz se evalúa como letra terminada

    def wheelEvent(self, ev):
        pasos = ev.angleDelta().y() / 120.0
        nuevo = min(ZOOM_MAX, max(ZOOM_MIN, self.zoom * (1.12 ** pasos)))
        if abs(nuevo - self.zoom) > 1e-6:
            self.zoom = nuevo
            self.update()
            self.trazos_cambiaron.emit()

    # ---------------------------------------------------------------- OpenGL
    def initializeGL(self):
        self._prog_esfera = self._programa(FRAG_ESFERA)
        self._prog_malla = self._programa(FRAG_MALLA)
        self._vb_esfera, self._n_esfera = self._buffer(_malla_esfera())
        self._vb_malla, self._n_malla = self._buffer(_lineas_malla())
        self._listo_gl = True

    def _programa(self, frag: str) -> QOpenGLShaderProgram:
        prog = QOpenGLShaderProgram(self)
        prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, VERT)
        prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, frag)
        prog.bindAttributeLocation("posicion", 0)
        prog.link()
        return prog

    def _buffer(self, datos: np.ndarray):
        vb = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        vb.create()
        vb.bind()
        crudo = datos.tobytes()
        vb.allocate(crudo, len(crudo))
        vb.release()
        return vb, len(datos)

    def _matrices(self):
        """(mvp, modelo): proyección ortográfica en la que la esfera unitaria mide `radio()`
        píxeles, por la rotación de la esfera."""
        r = self.radio()
        w, h = self.width() / r, self.height() / r
        proy = QMatrix4x4()
        proy.ortho(-w / 2, w / 2, -h / 2, h / 2, -10.0, 10.0)
        R = self.rotacion
        modelo = QMatrix4x4(*(float(R[i, j]) if i < 3 and j < 3 else float(i == j)
                              for i in range(4) for j in range(4)))
        return proy * modelo, modelo

    def _dibujar(self, prog, vb, n, modo):
        prog.bind()
        mvp, modelo = self._matrices()
        prog.setUniformValue("mvp", mvp)
        prog.setUniformValue("modelo", modelo)
        vb.bind()
        prog.enableAttributeArray(0)
        prog.setAttributeBuffer(0, GL_FLOAT, 0, 3, 0)
        self.context().functions().glDrawArrays(modo, 0, n)
        prog.disableAttributeArray(0)
        vb.release()
        prog.release()

    def paintGL(self):
        f = self.context().functions()
        fondo = T.color(T.SUPERFICIE)
        f.glClearColor(fondo.redF(), fondo.greenF(), fondo.blueF(), 1.0)
        f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self._listo_gl:
            f.glEnable(GL_DEPTH_TEST)
            f.glDepthFunc(GL_LEQUAL)
            f.glEnable(GL_MULTISAMPLE)
            self._prog_esfera.bind()
            self._prog_esfera.setUniformValue("color_base", *_rgb("#f4f8fb"))
            self._prog_esfera.setUniformValue("color_borde", *_rgb(T.GUIA_FUERTE))
            self._dibujar(self._prog_esfera, self._vb_esfera, self._n_esfera, GL_TRIANGLES)
            f.glEnable(GL_BLEND)
            f.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            self._prog_malla.bind()
            g = T.color(T.GUIA_FUERTE)
            self._prog_malla.setUniformValue("color", g.redF(), g.greenF(), g.blueF(), 0.55)
            self._dibujar(self._prog_malla, self._vb_malla, self._n_malla, GL_LINES)
            f.glDisable(GL_BLEND)
            f.glDisable(GL_DEPTH_TEST)
        self._pintar_encima()

    # ------------------------------------------------- capa 2D (QPainter)
    def _pintar_encima(self):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._dibujar_guias(p)
        if self.vacio:
            self._dibujar_ayuda(p)
        else:
            self._dibujar_trazos(p)
            if self.proyeccion and not self.no_definido:
                self._dibujar_proyeccion(p)
            self._dibujar_cursor(p)
        p.end()

    def _dibujar_guias(self, p: QPainter):
        """Las líneas guía del 2D, fijas en pantalla: arriba siempre es arriba."""
        from src.FeatureExtractor import ANGULO_LIENZO
        r, c = self.radio(), self.centro()
        p.setFont(T.fuente(11))
        for nombre, fy in T.LINEAS_GUIA:
            w = -(fy - 0.5) * ANGULO_LIENZO / self.zoom       # ángulo desde el centro de la vista
            if abs(w) >= math.pi / 2:
                continue
            y = c.y() - math.sin(w) * r
            media = math.cos(w) * r                            # media cuerda de la esfera a esa altura
            x0, x1 = max(16.0, c.x() - media), min(self.width() - 16.0, c.x() + media)
            if x1 <= x0:
                continue
            if nombre == "línea base":
                p.setPen(QPen(T.color(T.GUIA_FUERTE), 2))
            else:
                pluma = QPen(T.color(T.GUIA_FUERTE, 0.9), 1.5, Qt.PenStyle.CustomDashLine)
                pluma.setDashPattern([4, 4])
                p.setPen(pluma)
            p.drawLine(QPointF(x0, y), QPointF(x1, y))
            p.setPen(T.color(T.APAGADO))
            p.drawText(QPointF(x0 + 6, y - 6), nombre)

    def _dibujar_ayuda(self, p: QPainter):
        p.setPen(T.color(T.APAGADO))
        p.setFont(T.fuente(13))
        p.drawText(QRectF(0, self.height() - 34, self.width(), 24), Qt.AlignmentFlag.AlignCenter,
                   "Clic izquierdo: dibujar · clic derecho: girar · rueda: acercar")

    def _caminos_visibles(self, trazo):
        """Partes visibles (z > 0 en la vista) de un trazo del marco de la esfera, en pantalla."""
        R = self.rotacion
        caminos, actual = [], []
        for q in trazo:
            v = R @ np.asarray(q)
            if v[2] > 0:
                actual.append(self._pantalla(v))
            elif actual:
                caminos.append(actual)
                actual = []
        if actual:
            caminos.append(actual)
        return caminos

    def _trazar(self, p: QPainter, puntos):
        if len(puntos) == 1:
            p.drawPoint(puntos[0])
            return
        camino = QPainterPath(puntos[0])
        for q in puntos[1:]:
            camino.lineTo(q)
        p.drawPath(camino)

    def _dibujar_trazos(self, p: QPainter):
        pluma = QPen(T.color(T.TINTA), max(3.5, self.radio() / 50))
        pluma.setCapStyle(Qt.PenCapStyle.RoundCap)
        pluma.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pluma)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for trazo in self.trazos:
            for camino in self._caminos_visibles(trazo):
                self._trazar(p, camino)

    def _dibujar_proyeccion(self, p: QPainter):
        pluma = QPen(T.color(T.TEAL_TRAZO), 5)
        pluma.setCapStyle(Qt.PenCapStyle.RoundCap)
        pluma.setStyle(Qt.PenStyle.CustomDashLine)
        pluma.setDashPattern([0.01, 2.8])
        p.setPen(pluma)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for tramo in self.proyeccion:
            for camino in self._caminos_visibles(tramo):
                if len(camino) > 1:
                    self._trazar(p, camino)

    def _dibujar_cursor(self, p: QPainter):
        if not self._dibujando and not self.proyeccion:
            return
        v = self.rotacion @ np.asarray(self.trazos[-1][-1])
        if v[2] <= 0:
            return
        centro = self._pantalla(v)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(T.color(T.PRIMARIO, 0.15))
        p.drawEllipse(centro, 18, 18)
        p.setBrush(T.color(T.PRIMARIO))
        p.drawEllipse(centro, 9, 9)
