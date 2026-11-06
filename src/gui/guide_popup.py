"""
guide_popup.py
==============
«Guía de letras»: panel desplegable que muestra cómo se dibuja cada letra, en
la misma tipografía de Glifo Capture y sobre las mismas líneas guía del lienzo.

El número de trazos de cada letra sale del dataset del equipo (la mediana de
trazos por muestra), no de una regla fija.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QFontMetricsF, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import theme as T
from .widgets import dibujar_letra, etiqueta

RUTA_DATASET = T.RAIZ / "data"


def trazos_por_letra() -> dict:
    """{letra: trazos habituales} según el dataset más reciente (vacío si no hay)."""
    candidatos = sorted(Path(RUTA_DATASET).glob("glifo-dataset-*.json"))
    if not candidatos:
        return {}
    try:
        with open(candidatos[-1], encoding="utf-8") as f:
            datos = json.load(f)
        return {l: round(statistics.median(len(m) for m in ms)) for l, ms in datos.items() if ms}
    except (OSError, ValueError, statistics.StatisticsError):
        return {}


class VistaPrevia(QWidget):
    """La letra en grande sobre las cuatro líneas guía."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(250, 312)
        self.letra = "a"

    def fijar(self, letra: str):
        self.letra = letra
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(T.color(T.BORDE), 1))
        p.setBrush(T.color(T.FONDO))
        p.drawRoundedRect(r, 12, 12)
        ys = {}
        p.setFont(T.fuente(10))
        for nombre, fy in T.LINEAS_GUIA:
            y = fy * self.height()
            ys[nombre] = y
            if nombre == "línea base":
                p.setPen(QPen(T.color(T.GUIA_FUERTE), 1.5))
            else:
                pl = QPen(T.color(T.GUIA_FUERTE), 1, Qt.PenStyle.CustomDashLine)
                pl.setDashPattern([4, 4])
                p.setPen(pl)
            p.drawLine(QPointF(14, y), QPointF(self.width() - 14, y))
            p.setPen(T.color(T.APAGADO))
            p.drawText(QPointF(16, y - 4), nombre)
        # Tamaño de letra tal que su altura x ocupe justo el espacio entre «altura x» y «línea base».
        altura_x = ys["línea base"] - ys["altura x"]
        xh_100 = QFontMetricsF(T.fuente(100)).xHeight()
        tam = 100 * altura_x / xh_100
        dibujar_letra(p, self.letra, QRectF(self.rect()), tam, T.TINTA, base=ys["línea base"])


class BotonLetra(QPushButton):
    def __init__(self, letra: str, parent=None):
        super().__init__(parent)
        self.letra = letra
        self.setCheckable(True)
        self.setFixedHeight(46)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("QPushButton { border: none; background: transparent; }")

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.75, 0.75, -0.75, -0.75)
        sel = self.isChecked()
        p.setPen(QPen(T.color(T.PRIMARIO), 1.5) if sel else Qt.PenStyle.NoPen)
        p.setBrush(T.color(T.SUPERFICIE if sel else T.CHIP))
        p.drawRoundedRect(r, 8, 8)
        dibujar_letra(p, self.letra, QRectF(self.rect()), 20, T.TINTA, base=27)


class PanelGuia(QFrame):
    cerrar = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("panelGuia")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"#panelGuia {{ background: {T.SUPERFICIE}; border: 1px solid {T.BORDE}; border-radius: 16px; }}")
        self.setFixedWidth(560)
        sombra = QGraphicsDropShadowEffect(self)
        sombra.setBlurRadius(50)
        sombra.setOffset(0, 18)
        sombra.setColor(T.color(T.TINTA, 0.22))
        self.setGraphicsEffect(sombra)
        self._trazos = trazos_por_letra()

        caja = QVBoxLayout(self)
        caja.setContentsMargins(22, 22, 22, 22)
        caja.setSpacing(16)

        cabecera = QHBoxLayout()
        titulos = QVBoxLayout()
        titulos.setSpacing(4)
        titulos.addWidget(etiqueta("Cómo se dibuja cada letra", 18, T.TINTA, True))
        titulos.addWidget(etiqueta("Dibuja siguiendo este modelo. Es la misma letra de Glifo Capture.", 13, T.APAGADO))
        cabecera.addLayout(titulos, 1)
        x = QPushButton("✕")
        x.setFixedSize(32, 32)
        x.setCursor(Qt.CursorShape.PointingHandCursor)
        x.setToolTip("Cerrar la guía")
        x.setStyleSheet(f"QPushButton {{ border: 1px solid {T.BORDE}; border-radius: 8px; color: {T.APAGADO};"
                        f" background: {T.SUPERFICIE}; font-size: 13px; }} QPushButton:hover {{ border-color: {T.PRIMARIO}; }}")
        x.clicked.connect(self.cerrar.emit)
        cabecera.addWidget(x, 0, Qt.AlignmentFlag.AlignTop)
        caja.addLayout(cabecera)

        cuerpo = QHBoxLayout()
        cuerpo.setSpacing(20)
        izq = QVBoxLayout()
        izq.setSpacing(8)
        self.vista = VistaPrevia()
        izq.addWidget(self.vista)
        self.leyenda = QLabel()
        self.leyenda.setTextFormat(Qt.TextFormat.RichText)
        self.leyenda.setFont(T.fuente(12))
        self.leyenda.setStyleSheet(f"color: {T.APAGADO}; background: transparent;")
        self.leyenda.setFixedWidth(250)
        self.leyenda.setWordWrap(True)
        izq.addWidget(self.leyenda)
        izq.addStretch(1)
        cuerpo.addLayout(izq)

        der = QVBoxLayout()
        der.setSpacing(14)
        rejilla = QGridLayout()
        rejilla.setSpacing(6)
        self.botones = {}
        for i, letra in enumerate(T.LETRAS):
            b = BotonLetra(letra)
            b.clicked.connect(lambda _=False, l=letra: self.seleccionar(l))
            rejilla.addWidget(b, i // 6, i % 6)
            self.botones[letra] = b
        der.addLayout(rejilla)
        nota = QLabel("✎  <b>Tamaño:</b> el cuerpo de la letra va entre la altura x y la línea base; "
                      "las partes altas suben hasta la ascendente y las colas bajan hasta la descendente.")
        nota.setWordWrap(True)
        nota.setFont(T.fuente(13))
        nota.setStyleSheet("background: #eef6f3; color: #2f5d52; border-radius: 10px; padding: 10px 12px;")
        der.addWidget(nota)
        der.addStretch(1)
        cuerpo.addLayout(der, 1)
        caja.addLayout(cuerpo)
        self.seleccionar("a")

    def seleccionar(self, letra: str):
        for l, b in self.botones.items():
            b.setChecked(l == letra)
        self.vista.fijar(letra)
        n = self._trazos.get(letra)
        detalle = f" · se dibuja en {n} {'trazo' if n == 1 else 'trazos'}" if n else ""
        self.leyenda.setText(f"Letra «{T.html_letra(letra)}»{detalle}")


class CapaGuia(QWidget):
    """Capa sobre la ventana: oscurece el fondo y muestra el panel bajo el botón.

    Se cierra con la ✕, con el botón «Guía de letras» o haciendo clic fuera del panel.
    """

    cerrada = Signal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.panel = PanelGuia(self)
        self.panel.cerrar.connect(self.ocultar)
        self.ancla_derecha = 0   # borde derecho del botón, en coordenadas de la capa
        self.hide()

    def mostrar(self, letra: str | None = None):
        if letra:
            self.panel.seleccionar(letra)
        self.setGeometry(self.parentWidget().rect().adjusted(0, 72, 0, 0))
        self.panel.adjustSize()
        x = max(16, self.ancla_derecha - self.panel.width() + 60)
        self.panel.move(x, 8)
        self.show()
        self.raise_()

    def ocultar(self):
        if self.isVisible():
            self.hide()
            self.cerrada.emit()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), T.color(T.TINTA, 0.18))
        # Pico que apunta al botón
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx = self.ancla_derecha - 70
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(T.color(T.SUPERFICIE))
        p.drawPolygon([QPointF(cx - 9, 9), QPointF(cx, 0), QPointF(cx + 9, 9)])

    def mousePressEvent(self, ev):
        if not self.panel.geometry().contains(ev.position().toPoint()):
            self.ocultar()
