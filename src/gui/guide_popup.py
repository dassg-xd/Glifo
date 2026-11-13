"""
guide_popup.py
==============
«Guía de letras»: panel desplegable que muestra cómo se traza cada letra, como
la guía de dibujo de Glifo Capture: la letra completa en gris y, encima, cada
trazo se dibuja solo y en orden, con su número donde empieza.

Los trazos son los de la guía de dibujo de Glifo Capture, hechos a mano
(data/guia_dibujo.json): curvas limpias y simétricas, solo para mostrar.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from PySide6.QtCore import QElapsedTimer, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QPainterPath, QPen
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
from .guia_trazos import GUIA_DIBUJO, puntos_de_camino
from .widgets import dibujar_letra, etiqueta

# Ventana del espacio 0..1000 que se muestra (la misma de Glifo Capture): x 200..800, y 130..930.
VISTA_X, VISTA_Y, VISTA_ANCHO, VISTA_ALTO = 200.0, 130.0, 600.0, 800.0
ESPERA_INICIAL, PAUSA_ENTRE_TRAZOS = 250, 300      # milisegundos
MS_POR_UNIDAD, DURACION_MINIMA = 1.7, 450          # duración de cada trazo según su largo

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
    """Las cuatro líneas guía, la letra completa en gris y la animación de sus trazos."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(250, 312)
        self.letra = "a"
        self._trazos: list[list[tuple]] = []      # puntos en el espacio 0..1000
        self._largos: list[float] = []
        self._inicios: list[float] = []           # ms en que empieza cada trazo
        self._numeros: list[QPointF] = []
        self._reloj = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._avanzar)

    # --------------------------------------------------------------- datos
    def fijar(self, letra: str):
        self.letra = letra
        self._trazos = [puntos_de_camino(c, 3.0) for c in GUIA_DIBUJO.get(letra, [])]
        self._largos = [sum(math.dist(a, b) for a, b in zip(t, t[1:])) for t in self._trazos]
        self._inicios, t = [], ESPERA_INICIAL
        for largo in self._largos:
            self._inicios.append(t)
            t += (250 if largo < 5 else max(DURACION_MINIMA, largo * MS_POR_UNIDAD)) + PAUSA_ENTRE_TRAZOS
        self._numeros = self._ubicar_numeros()
        self.repetir()

    @property
    def n_trazos(self) -> int:
        return len(self._trazos)

    def repetir(self):
        self._reloj.restart()
        self._timer.start()
        self.update()

    def _avanzar(self):
        if not self._inicios or self._reloj.elapsed() > self._inicios[-1] + self._duracion(-1) + 100:
            self._timer.stop()
        self.update()

    def _duracion(self, i: int) -> float:
        largo = self._largos[i]
        return 250 if largo < 5 else max(DURACION_MINIMA, largo * MS_POR_UNIDAD)

    # ---------------------------------------------------------- geometría
    def _escala(self) -> float:
        return min(self.width() / VISTA_ANCHO, self.height() / VISTA_ALTO)

    def _a_pantalla(self, p) -> QPointF:
        e = self._escala()
        dx = (self.width() - VISTA_ANCHO * e) / 2
        return QPointF(dx + (p[0] - VISTA_X) * e, (p[1] - VISTA_Y) * e)

    def _ubicar_numeros(self) -> list:
        """Cada número va junto al inicio de su trazo, hacia atrás de la dirección en que
        arranca para no taparlo; si choca con uno anterior, prueba a los lados y más lejos."""
        puestos = []
        for t in self._trazos:
            a = t[0]
            b = next((q for q in t if math.dist(q, a) >= 40), t[-1])
            dx, dy = a[0] - b[0], a[1] - b[1]
            m = math.hypot(dx, dy)
            dx, dy = (-1.0, 0.0) if m < 1 else (dx / m, dy / m)
            for ux, uy, dist in ((dx, dy, 58), (-dy, dx, 58), (dy, -dx, 58), (dx, dy, 120), (-dy, dx, 120), (dy, -dx, 120)):
                cx = min(VISTA_X + VISTA_ANCHO - 35, max(VISTA_X + 35, a[0] + ux * dist))
                cy = min(VISTA_Y + VISTA_ALTO - 25, max(VISTA_Y + 35, a[1] + uy * dist))
                if all(math.dist((cx, cy), q) > 76 for q in puestos):
                    break
            puestos.append((cx, cy))
        return [self._a_pantalla(q) for q in puestos]

    # --------------------------------------------------------------- dibujo
    def _camino(self, puntos) -> QPainterPath:
        camino = QPainterPath(self._a_pantalla(puntos[0]))
        for q in puntos[1:]:
            camino.lineTo(self._a_pantalla(q))
        return camino

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(T.color(T.BORDE), 1))
        p.setBrush(T.color(T.FONDO))
        p.drawRoundedRect(r, 12, 12)
        p.setFont(T.fuente(10))
        for nombre, fy in T.LINEAS_GUIA:
            y = self._a_pantalla((0, fy * 1000)).y()
            if nombre == "línea base":
                p.setPen(QPen(T.color(T.GUIA_FUERTE), 1.5))
            else:
                pl = QPen(T.color(T.GUIA_FUERTE), 1, Qt.PenStyle.CustomDashLine)
                pl.setDashPattern([4, 4])
                p.setPen(pl)
            p.drawLine(QPointF(14, y), QPointF(self.width() - 14, y))
            p.setPen(T.color(T.APAGADO))
            p.drawText(QPointF(16, y - 4), nombre)
        if not self._trazos:          # sin guía: la letra en la tipografía, como antes
            dibujar_letra(p, self.letra, QRectF(self.rect()), 110, T.TINTA,
                          base=self._a_pantalla((0, 700)).y())
            return

        ancho = 26 * self._escala()

        def pluma(color):
            q = QPen(T.color(color), ancho)
            q.setCapStyle(Qt.PenCapStyle.RoundCap)
            q.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            return q
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(pluma(T.GUIA))                                   # la letra completa, de fondo
        for t in self._trazos:
            if len(t) > 1:
                p.drawPath(self._camino(t))
            else:
                p.drawPoint(self._a_pantalla(t[0]))

        ahora = self._reloj.elapsed() if self._reloj.isValid() else 1e9
        p.setPen(pluma(T.TINTA))
        for i, t in enumerate(self._trazos):
            avance = (ahora - self._inicios[i]) / self._duracion(i)
            if avance <= 0:
                break
            avance = min(1.0, avance)
            avance = avance * avance * (3 - 2 * avance)            # suave al empezar y al terminar
            if len(t) < 2 or self._largos[i] < 5:
                p.drawPoint(self._a_pantalla(t[0]))                 # un punto
                continue
            meta, recorrido, parcial = avance * self._largos[i], 0.0, [t[0]]
            for a, b in zip(t, t[1:]):
                d = math.dist(a, b)
                if recorrido + d >= meta:
                    f = (meta - recorrido) / d if d else 0
                    parcial.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
                    break
                parcial.append(b)
                recorrido += d
            p.drawPath(self._camino(parcial))

        p.setFont(T.fuente(11, True))
        radio = 34 * self._escala()
        for i, c in enumerate(self._numeros):
            if ahora < self._inicios[i]:
                break
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(T.color(T.TEAL_TRAZO))
            p.drawEllipse(c, radio, radio)
            p.setPen(T.color("#ffffff"))
            p.drawText(QRectF(c.x() - radio, c.y() - radio, 2 * radio, 2 * radio), Qt.AlignmentFlag.AlignCenter, str(i + 1))


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
        titulos.addWidget(etiqueta("Sigue el orden de los números: es la misma guía de Glifo Capture.", 13, T.APAGADO))
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
        self.boton_repetir = QPushButton("↻ Repetir")
        self.boton_repetir.setCursor(Qt.CursorShape.PointingHandCursor)
        self.boton_repetir.setToolTip("Vuelve a mostrar el orden de los trazos")
        self.boton_repetir.setStyleSheet(f"QPushButton {{ border: none; background: transparent; color: {T.PRIMARIO};"
                                         f" font-weight: bold; font-size: 13px; text-align: left; padding: 0; }}")
        self.boton_repetir.clicked.connect(lambda: self.vista.repetir())
        izq.addWidget(self.boton_repetir, 0, Qt.AlignmentFlag.AlignLeft)
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
        n = self.vista.n_trazos or self._trazos.get(letra)
        if n == 1:
            detalle = "1 trazo: se dibuja de una vez, sin levantar el lápiz, empezando en el 1."
        elif n:
            detalle = f"{n} trazos: empieza en el 1 y levanta el lápiz entre uno y otro, en orden."
        else:
            detalle = ""
        self.leyenda.setText(f"Letra «{T.html_letra(letra)}» · {detalle}" if detalle else f"Letra «{T.html_letra(letra)}»")


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
