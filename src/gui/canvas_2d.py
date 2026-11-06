"""
canvas_2d.py
============
Lienzo 2D: captura el trazo con el ratón (o el dedo) y dibuja encima la
proyección punteada de la letra más probable.

Los puntos se guardan normalizados (x/L, y/L, con L = lado del cuadrado de
dibujo), que es lo que espera el FeatureExtractor. El lienzo siempre es
cuadrado y se centra en el espacio disponible.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from . import theme as T

LADO_MAXIMO = 560
DISTANCIA_MINIMA = 0.0015   # en fracción del lienzo: filtra eventos repetidos del ratón


class Lienzo2D(QWidget):
    trazos_cambiaron = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(360, 360)
        self.setMouseTracking(False)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.trazos: list[list[tuple[float, float]]] = []
        self.proyeccion: list[tuple[float, float]] = []
        self.no_definido = False
        self._dibujando = False

    # ------------------------------------------------------------ geometría
    def cuadro(self) -> QRectF:
        lado = min(self.width(), self.height(), LADO_MAXIMO)
        return QRectF((self.width() - lado) / 2, (self.height() - lado) / 2, lado, lado)

    def _a_normal(self, pos: QPointF):
        c = self.cuadro()
        x = (pos.x() - c.left()) / c.width()
        y = (pos.y() - c.top()) / c.height()
        return (min(1.0, max(0.0, x)), min(1.0, max(0.0, y)))

    def _a_pantalla(self, p) -> QPointF:
        c = self.cuadro()
        return QPointF(c.left() + p[0] * c.width(), c.top() + p[1] * c.height())

    # ------------------------------------------------------------- acciones
    def limpiar(self) -> None:
        self.trazos = []
        self.proyeccion = []
        self.no_definido = False
        self._dibujando = False
        self.update()
        self.trazos_cambiaron.emit()

    def mostrar_resultado(self, proyeccion, no_definido: bool) -> None:
        self.proyeccion = list(proyeccion)
        self.no_definido = no_definido
        self.update()

    @property
    def vacio(self) -> bool:
        return not any(self.trazos)

    # -------------------------------------------------------------- eventos
    def mousePressEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton or not self.cuadro().contains(ev.position()):
            return
        self._dibujando = True
        self.trazos.append([self._a_normal(ev.position())])
        self.update()
        self.trazos_cambiaron.emit()

    def mouseMoveEvent(self, ev):
        if not self._dibujando:
            return
        p = self._a_normal(ev.position())
        u = self.trazos[-1][-1]
        if abs(p[0] - u[0]) + abs(p[1] - u[1]) >= DISTANCIA_MINIMA:
            self.trazos[-1].append(p)
            self.update()
            self.trazos_cambiaron.emit()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and self._dibujando:
            self._dibujando = False
            self.update()

    # -------------------------------------------------------------- dibujo
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = self.cuadro()

        # Borde punteado del área de dibujo
        borde = QPen(T.color(T.GUIA), 1.5, Qt.PenStyle.DashLine)
        borde.setDashPattern([4, 4])
        p.setPen(borde)
        p.setBrush(T.color(T.SUPERFICIE))
        p.drawRoundedRect(c.adjusted(0.75, 0.75, -0.75, -0.75), 16, 16)

        self._dibujar_guias(p, c)
        if self.vacio:
            self._dibujar_estado_vacio(p, c)
        else:
            self._dibujar_trazos(p, c)
            if self.proyeccion and not self.no_definido:
                self._dibujar_proyeccion(p)
            self._dibujar_cursor(p)
            if self.no_definido:
                self._dibujar_aviso(p, c)

    def _dibujar_guias(self, p: QPainter, c: QRectF):
        p.setFont(T.fuente(11))
        for nombre, fy in T.LINEAS_GUIA:
            y = c.top() + fy * c.height()
            if nombre == "línea base":
                p.setPen(QPen(T.color(T.GUIA_FUERTE), 2))
            else:
                pluma = QPen(T.color(T.GUIA), 1.5, Qt.PenStyle.CustomDashLine)
                pluma.setDashPattern([4, 4])
                p.setPen(pluma)
            p.drawLine(QPointF(c.left() + 16, y), QPointF(c.right() - 16, y))
            p.setPen(T.color(T.APAGADO))
            p.drawText(QPointF(c.left() + 20, y - 6), nombre)

    def _dibujar_estado_vacio(self, p: QPainter, c: QRectF):
        caja = QRectF(c.left() + c.width() * 0.196, c.top() + c.height() * 0.446,
                      c.width() * 0.608, 130)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(T.color(T.TINTA, 0.06))
        p.drawRoundedRect(caja.translated(0, 2), 20, 20)
        p.setBrush(T.color(T.SUPERFICIE))
        p.drawRoundedRect(caja, 20, 20)
        # logo pequeño
        p.save()
        p.translate(caja.center().x() - 26, caja.top() + 16)
        p.scale(52 / 34, 52 / 34)
        pluma = QPen(T.color(T.PRIMARIO), 2.2)
        pluma.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pluma)
        camino = QPainterPath(QPointF(3, 21))
        camino.cubicTo(6, 9, 10, 3, 13, 4)
        camino.cubicTo(16, 5, 13, 16, 12, 19)
        camino.lineTo(21, 6)
        p.drawPath(camino)
        pluma.setColor(T.color(T.TEAL_TRAZO))
        p.setPen(pluma)
        p.drawLine(QPointF(25, 5), QPointF(30, 5))
        p.restore()
        p.setPen(T.color(T.TINTA))
        p.setFont(T.fuente(19, True))
        p.drawText(QRectF(caja.left(), caja.top() + 60, caja.width(), 28),
                   Qt.AlignmentFlag.AlignCenter, "Dibuja una letra aquí")
        p.setPen(T.color(T.APAGADO))
        p.setFont(T.fuente(13))
        p.drawText(QRectF(caja.left(), caja.top() + 90, caja.width(), 22),
                   Qt.AlignmentFlag.AlignCenter, "Una letra de tamaño medio se reconoce mejor")

    def _dibujar_trazos(self, p: QPainter, c: QRectF):
        pluma = QPen(T.color(T.TINTA), max(3.5, c.width() / 112))
        pluma.setCapStyle(Qt.PenCapStyle.RoundCap)
        pluma.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pluma)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for trazo in self.trazos:
            if len(trazo) == 1:
                p.drawPoint(self._a_pantalla(trazo[0]))
                continue
            camino = QPainterPath(self._a_pantalla(trazo[0]))
            for q in trazo[1:]:
                camino.lineTo(self._a_pantalla(q))
            p.drawPath(camino)

    def _dibujar_proyeccion(self, p: QPainter):
        pluma = QPen(T.color(T.TEAL_TRAZO), 5)
        pluma.setCapStyle(Qt.PenCapStyle.RoundCap)
        pluma.setStyle(Qt.PenStyle.CustomDashLine)
        pluma.setDashPattern([0.01, 2.8])
        p.setPen(pluma)
        camino = QPainterPath(self._a_pantalla(self.proyeccion[0]))
        for q in self.proyeccion[1:]:
            camino.lineTo(self._a_pantalla(q))
        p.drawPath(camino)

    def _dibujar_cursor(self, p: QPainter):
        if not self._dibujando and not self.proyeccion:
            return
        centro = self._a_pantalla(self.trazos[-1][-1])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(T.color(T.PRIMARIO, 0.15))
        p.drawEllipse(centro, 18, 18)
        p.setBrush(T.color(T.PRIMARIO))
        p.drawEllipse(centro, 9, 9)

    def _dibujar_aviso(self, p: QPainter, c: QRectF):
        texto = "Trazo no definido"
        f = T.fuente(13, True)
        p.setFont(f)
        ancho = p.fontMetrics().horizontalAdvance(texto) + 54
        caja = QRectF(c.center().x() - ancho / 2, c.top() + 18, ancho, 32)
        p.setPen(QPen(T.color(T.AVISO_BORDE), 1))
        p.setBrush(T.color("#fdf3e1"))
        p.drawRoundedRect(caja, 16, 16)
        # triángulo
        p.save()
        p.translate(caja.left() + 14, caja.top() + 8)
        p.scale(16 / 24, 16 / 24)
        pt = QPen(T.color(T.AVISO_ICONO), 2.2)
        pt.setCapStyle(Qt.PenCapStyle.RoundCap)
        pt.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pt)
        p.setBrush(Qt.BrushStyle.NoBrush)
        tri = QPainterPath(QPointF(12, 3))
        tri.lineTo(2.5, 20)
        tri.lineTo(21.5, 20)
        tri.closeSubpath()
        p.drawPath(tri)
        p.drawLine(QPointF(12, 10), QPointF(12, 14.5))
        p.drawPoint(QPointF(12, 17.5))
        p.restore()
        p.setPen(T.color("#8a5a12"))
        p.drawText(QRectF(caja.left() + 38, caja.top(), ancho - 44, 32),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, texto)
