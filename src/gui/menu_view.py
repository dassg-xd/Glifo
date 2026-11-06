"""
menu_view.py
============
Ventana A: menú principal. Permite elegir el modo de trabajo (2D o 3D) y salir.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from . import theme as T
from .widgets import Logo, Tarjeta, boton, etiqueta


class IconoModo(QWidget):
    """Ícono de las tarjetas: un lienzo con trazo (2D) o una esfera (3D)."""

    def __init__(self, modo: str, parent=None):
        super().__init__(parent)
        self.modo = modo
        self.setFixedSize(80, 80)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.modo == "2D":
            p.setPen(QPen(T.color(T.PRIMARIO), 2.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(QRectF(4, 4, 72, 72), 10, 10)
            pluma = QPen(T.color(T.PRIMARIO), 3)
            pluma.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pluma)
            c = QPainterPath(QPointF(20, 56))
            c.cubicTo(24, 40, 30, 28, 36, 30)
            c.cubicTo(42, 32, 36, 46, 35, 52)
            c.lineTo(56, 26)
            p.drawPath(c)
        else:
            p.setPen(QPen(T.color(T.TEAL), 2.2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QRectF(2, 2, 76, 76))
            p.drawEllipse(QRectF(22, 2, 36, 76))
            p.drawEllipse(QRectF(2, 26, 76, 28))


class TarjetaModo(Tarjeta):
    entrar = Signal()

    def __init__(self, modo: str, subtitulo: str, descripcion: str, color: str, parent=None):
        super().__init__(parent)
        self.setFixedSize(380, 250)
        caja = QVBoxLayout(self)
        caja.setContentsMargins(34, 26, 34, 18)
        caja.setSpacing(0)

        arriba = QHBoxLayout()
        arriba.setSpacing(22)
        arriba.addWidget(IconoModo(modo), 0, Qt.AlignmentFlag.AlignTop)
        titulos = QVBoxLayout()
        titulos.setSpacing(0)
        titulos.addWidget(etiqueta(modo, 52, color, True))
        titulos.addWidget(etiqueta(subtitulo, 17, T.TINTA, True))
        titulos.addStretch(1)
        arriba.addLayout(titulos)
        arriba.addStretch(1)
        caja.addLayout(arriba)
        caja.addSpacing(26)

        texto = etiqueta(descripcion, 14, T.APAGADO)
        texto.setWordWrap(True)
        caja.addWidget(texto)
        caja.addStretch(1)

        b = boton("Entrar", "primario", alto=32)
        b.setStyleSheet(
            f"QPushButton {{ background: {color}; border-color: {color}; font-size: 14px; }}"
            f"QPushButton:hover {{ background: {T.TINTA}; border-color: {T.TINTA}; }}"
        )
        b.clicked.connect(self.entrar.emit)
        caja.addWidget(b)


class Paso(QWidget):
    def __init__(self, numero: int, texto: str, parent=None):
        super().__init__(parent)
        fila = QHBoxLayout(self)
        fila.setContentsMargins(0, 0, 0, 0)
        fila.setSpacing(14)
        circulo = QLabel(str(numero))
        circulo.setFixedSize(26, 26)
        circulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        circulo.setFont(T.fuente(12, True))
        circulo.setStyleSheet(f"background: {T.TEAL_SUAVE}; color: {T.TEAL}; border-radius: 13px;")
        fila.addWidget(circulo)
        fila.addWidget(etiqueta(texto, 16, T.TEXTO))
        fila.addStretch(1)


class BotonSalir(QPushButton):
    """La [X] de la esquina: roja, cierra la aplicación."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("salir")
        self.setFixedSize(44, 44)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Salir de Glifo")

    def paintEvent(self, ev):
        super().paintEvent(ev)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pluma = QPen(Qt.GlobalColor.white, 2.4)
        pluma.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pluma)
        m = 14
        p.drawLine(QPointF(m, m), QPointF(44 - m, 44 - m))
        p.drawLine(QPointF(44 - m, m), QPointF(m, 44 - m))


class VistaMenu(QWidget):
    """Ventana A: bienvenida, elección de modo y salida."""

    elegir_modo = Signal(str)
    salir = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("fondo")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        columna = QVBoxLayout(self)
        columna.setContentsMargins(40, 36, 40, 34)
        columna.setSpacing(0)

        columna.addWidget(Logo(150, punteado=True), 0, Qt.AlignmentFlag.AlignHCenter)
        columna.addSpacing(26)
        titulo = etiqueta("Glifo", 64, T.TINTA, True)
        columna.addWidget(titulo, 0, Qt.AlignmentFlag.AlignHCenter)
        columna.addSpacing(6)
        columna.addWidget(
            etiqueta("Dibuja una letra y mira cómo el modelo la reconoce mientras la trazas", 18, T.APAGADO),
            0, Qt.AlignmentFlag.AlignHCenter,
        )
        columna.addSpacing(48)

        tarjetas = QHBoxLayout()
        tarjetas.setSpacing(20)
        tarjetas.addStretch(1)
        t2d = TarjetaModo("2D", "Plano", "Traza sobre un lienzo plano\ncon el ratón o con el dedo.", T.PRIMARIO)
        t3d = TarjetaModo("3D", "Esfera", "Traza sobre una esfera que\npuedes girar y acercar.", T.TEAL)
        t2d.entrar.connect(lambda: self.elegir_modo.emit("2D"))
        t3d.entrar.connect(lambda: self.elegir_modo.emit("3D"))
        tarjetas.addWidget(t2d)
        tarjetas.addWidget(t3d)
        tarjetas.addStretch(1)
        columna.addLayout(tarjetas)
        columna.addSpacing(40)

        como = Tarjeta()
        como.setFixedWidth(780)
        caja = QVBoxLayout(como)
        caja.setContentsMargins(36, 28, 36, 28)
        caja.setSpacing(14)
        caja.addWidget(etiqueta("Cómo funciona", 18, T.TINTA, True))
        caja.addSpacing(4)
        for i, texto in enumerate((
            "Dibuja una letra minúscula del alfabeto español.",
            "Mira a la derecha qué letras son más probables.",
            "Sigue la línea punteada: completa el trazo por ti.",
        ), start=1):
            caja.addWidget(Paso(i, texto))
        columna.addWidget(como, 0, Qt.AlignmentFlag.AlignHCenter)

        columna.addStretch(1)
        columna.addWidget(
            etiqueta("Universidad de los Llanos · Ingeniería de Sistemas · Procesos Estocásticos", 13, T.APAGADO),
            0, Qt.AlignmentFlag.AlignHCenter,
        )

        # La [X] va encima del contenido, fija en la esquina superior izquierda.
        self.boton_salir = BotonSalir(self)
        self.boton_salir.move(40, 32)
        self.boton_salir.clicked.connect(self.salir.emit)
        self.boton_salir.raise_()
