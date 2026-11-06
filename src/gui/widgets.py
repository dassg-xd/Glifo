"""
widgets.py
==========
Piezas visuales reutilizables: logo, tarjetas, fichas de letra, íconos y la
barra superior común a las ventanas 2D y 3D.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QFontMetricsF, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from . import theme as T


def _pluma(color: str, ancho: float, punteada: bool = False) -> QPen:
    p = QPen(T.color(color), ancho)
    p.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    if punteada:
        p.setStyle(Qt.PenStyle.CustomDashLine)
        p.setDashPattern([0.01, 2.6])
    return p


class Logo(QWidget):
    """El trazo de Glifo (una «n» manuscrita) con su remate teal.

    Con `punteado=True` el remate se dibuja como la proyección punteada del menú.
    """

    def __init__(self, ancho: int = 40, punteado: bool = False, parent=None):
        super().__init__(parent)
        self._punteado = punteado
        self.setFixedSize(ancho, round(ancho * (24 / 34 if not punteado else 0.45)))

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._punteado:
            # Versión del menú: trazo + cuatro puntos teal a la derecha.
            esc = self.width() / 150.0
            p.scale(esc, esc)
            camino = QPainterPath(QPointF(6, 58))
            camino.cubicTo(14, 24, 28, 2, 40, 6)
            camino.cubicTo(52, 10, 46, 40, 44, 54)
            camino.lineTo(84, 12)
            p.setPen(_pluma(T.PRIMARIO, 6))
            p.drawPath(camino)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(T.color(T.TEAL_TRAZO))
            for i in range(4):
                p.drawRoundedRect(QRectF(97 + i * 12, 9, 7, 5), 2, 2)
            return
        esc = self.width() / 34.0
        p.scale(esc, esc)
        camino = QPainterPath(QPointF(3, 21))
        camino.cubicTo(6, 9, 10, 3, 13, 4)
        camino.cubicTo(16, 5, 13, 16, 12, 19)
        camino.lineTo(21, 6)
        p.setPen(_pluma(T.PRIMARIO, 2.6))
        p.drawPath(camino)
        p.setPen(_pluma(T.TEAL_TRAZO, 2.6))
        p.drawLine(QPointF(25, 5), QPointF(30, 5))


class Tarjeta(QFrame):
    """Contenedor blanco con borde y esquinas redondeadas."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("tarjeta")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)


def etiqueta(texto: str, tam: float, color: str = T.TEXTO, negrita: bool = False, parent=None) -> QLabel:
    lab = QLabel(texto, parent)
    lab.setFont(T.fuente(tam, negrita))
    lab.setStyleSheet(f"color: {color}; background: transparent;")
    return lab


def boton(texto: str, tipo: str = "secundario", ancho: int | None = None, alto: int = 40) -> QPushButton:
    b = QPushButton(texto)
    b.setObjectName(tipo)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.setFixedHeight(alto)
    if ancho:
        b.setFixedWidth(ancho)
    return b


class FichaLetra(QWidget):
    """Una letra dentro de un recuadro redondeado.

    Todas las letras comparten la línea base: el cuerpo (altura x) queda
    centrado y las partes altas y las colas sobresalen, como en los mockups.
    """

    def __init__(self, letra: str, lado: int = 64, tam: float = 27, fondo: str = "#eef3f6",
                 color_letra: str = T.GRIS_LETRA, radio: float = 12, parent=None):
        super().__init__(parent)
        self.letra = letra
        self._tam = tam
        self._fondo = fondo
        self._color = color_letra
        self._radio = radio
        self._borde: str | None = None
        self.setFixedSize(lado, lado)

    def configurar(self, letra: str | None = None, fondo: str | None = None,
                   color_letra: str | None = None, borde: str | None = "sin cambio"):
        if letra is not None:
            self.letra = letra
        if fondo is not None:
            self._fondo = fondo
        if color_letra is not None:
            self._color = color_letra
        if borde != "sin cambio":
            self._borde = borde
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.75, 0.75, -0.75, -0.75)
        p.setPen(QPen(T.color(self._borde), 1.5) if self._borde else Qt.PenStyle.NoPen)
        p.setBrush(T.color(self._fondo))
        p.drawRoundedRect(r, self._radio, self._radio)
        dibujar_letra(p, self.letra, QRectF(self.rect()), self._tam, self._color)


def dibujar_letra(p: QPainter, letra: str, caja: QRectF, tam: float, color: str,
                  base: float | None = None) -> None:
    """Dibuja `letra` centrada en `caja`; con `base` se fija la línea base (en y)."""
    f = T.fuente_letra(letra, tam) if len(letra) == 1 else T.fuente(tam, negrita=True)
    p.setFont(f)
    p.setPen(T.color(color))
    fm = QFontMetricsF(f)
    ancho = fm.horizontalAdvance(letra)
    if base is None:
        xh = QFontMetricsF(T.fuente(tam)).xHeight()
        base = caja.center().y() + xh / 2
    p.drawText(QPointF(caja.center().x() - ancho / 2, base), letra)


class IconoAviso(QWidget):
    """Triángulo de advertencia (estado «Trazo no definido»)."""

    def __init__(self, lado: int = 16, parent=None):
        super().__init__(parent)
        self.setFixedSize(lado, lado)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        e = self.width() / 24.0
        p.scale(e, e)
        p.setPen(_pluma(T.AVISO_ICONO, 2.2))
        tri = QPainterPath(QPointF(12, 3))
        tri.lineTo(2.5, 20)
        tri.lineTo(21.5, 20)
        tri.closeSubpath()
        p.drawPath(tri)
        p.drawLine(QPointF(12, 10), QPointF(12, 14.5))
        p.drawPoint(QPointF(12, 17.5))


class IconoLibro(QWidget):
    """Ícono del botón «Guía de letras»."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(18, 18)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.scale(0.75, 0.75)
        p.setPen(_pluma(T.PRIMARIO, 2))
        c = QPainterPath(QPointF(4, 19.5))
        c.lineTo(4, 5)
        c.quadTo(4, 3, 6, 3)
        c.lineTo(20, 3)
        c.lineTo(20, 19)
        c.lineTo(6, 19)
        c.quadTo(4, 19, 4, 21)
        p.drawPath(c)
        p.drawLine(QPointF(8, 7), QPointF(16, 7))
        p.drawLine(QPointF(8, 11), QPointF(14, 11))


class Chevron(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.arriba = False
        self.setFixedSize(14, 14)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.scale(14 / 24, 14 / 24)
        p.setPen(_pluma(T.PRIMARIO, 2.4))
        if self.arriba:
            p.drawPolyline([QPointF(6, 15), QPointF(12, 9), QPointF(18, 15)])
        else:
            p.drawPolyline([QPointF(6, 9), QPointF(12, 15), QPointF(18, 9)])


class BotonGuia(QPushButton):
    """«Guía de letras» con ícono y flecha que cambia de dirección al abrirse."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("guia")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(40)
        fila = QHBoxLayout(self)
        fila.setContentsMargins(16, 0, 16, 0)
        fila.setSpacing(8)
        fila.addWidget(IconoLibro())
        texto = etiqueta("Guía de letras", 14, T.PRIMARIO, True)
        texto.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        fila.addWidget(texto)
        self._chevron = Chevron()
        fila.addWidget(self._chevron)
        self.setFixedWidth(fila.sizeHint().width() + 4)
        self.toggled.connect(self._al_cambiar)

    def _al_cambiar(self, abierto: bool):
        self._chevron.arriba = abierto
        self._chevron.update()


class BarraSuperior(QFrame):
    """Barra común de las ventanas 2D y 3D: logo, guía, menú y conmutador de modo."""

    pedir_menu = Signal()
    pedir_modo = Signal(str)  # "2D" o "3D"

    def __init__(self, modo: str, parent=None):
        super().__init__(parent)
        self.setObjectName("barra")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(72)
        fila = QHBoxLayout(self)
        fila.setContentsMargins(40, 0, 40, 0)
        fila.setSpacing(12)

        marca = QHBoxLayout()
        marca.setSpacing(12)
        marca.addWidget(Logo(40))
        marca.addWidget(etiqueta("Glifo", 23, T.TINTA, True))
        fila.addLayout(marca)
        fila.addStretch(1)

        self.boton_guia = BotonGuia()
        fila.addWidget(self.boton_guia)

        menu = boton("Menú", "secundario", 120)
        menu.clicked.connect(self.pedir_menu.emit)
        fila.addWidget(menu)

        conmutador = QFrame()
        conmutador.setFixedHeight(40)
        conmutador.setObjectName("conmutador")
        conmutador.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        caja = QHBoxLayout(conmutador)
        caja.setContentsMargins(4, 4, 4, 4)
        caja.setSpacing(4)
        grupo = QButtonGroup(self)
        self._botones_modo = {}
        for nombre, clave in (("Plano 2D", "2D"), ("Esfera 3D", "3D")):
            b = boton(nombre, "modo", alto=32)
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, c=clave: self.pedir_modo.emit(c))
            grupo.addButton(b)
            caja.addWidget(b)
            self._botones_modo[clave] = b
        fila.addWidget(conmutador, 0, Qt.AlignmentFlag.AlignVCenter)
        self.modo = modo
        self.fijar_modo(modo)

    def fijar_modo(self, modo: str) -> None:
        """Marca el modo propio de la vista (al volver a ella, el conmutador queda bien)."""
        self._botones_modo[modo].setChecked(True)


def espaciador_horizontal() -> QWidget:
    w = QWidget()
    w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    return w


__all__ = [
    "Logo", "Tarjeta", "etiqueta", "boton", "FichaLetra", "dibujar_letra", "IconoAviso",
    "BotonGuia", "BarraSuperior", "espaciador_horizontal", "QSize",
]
