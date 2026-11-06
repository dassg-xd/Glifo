"""
view_2d.py
==========
Ventana B: reconocedor y visualizador 2D.

Izquierda: lienzo con proyección punteada. Derecha: listado ordenado de
P(c | O_1:t). Abajo, centrados: «Limpiar lienzo» y «Reiniciar captura».
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from . import theme as T
from .canvas_2d import Lienzo2D
from .guide_popup import CapaGuia
from .ranking_panel import PanelClasificacion
from .recognizer import Reconocedor
from .widgets import BarraSuperior, Tarjeta, boton, etiqueta

SUB_VACIO = "Haz clic y arrastra para dibujar una letra"
SUB_DIBUJO = "Dibuja con el ratón; la proyección se actualiza en cada trazo"


class Leyenda(QWidget):
    """«Tu trazo» y «Proyección…» (o el aviso de que no hay proyección)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(20)
        self.no_definido = False

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        y = self.height() / 2
        p.setPen(QPen(T.color(T.TINTA), 4))
        p.drawLine(QPointF(0, y), QPointF(44, y))
        p.setFont(T.fuente(13))
        p.setPen(T.color(T.APAGADO))
        p.drawText(QPointF(54, y + 5), "Tu trazo")
        if self.no_definido:
            p.drawText(QPointF(170, y + 5), "Sin proyección: ninguna letra es lo bastante probable")
            return
        pl = QPen(T.color(T.TEAL_TRAZO), 4)
        pl.setCapStyle(Qt.PenCapStyle.RoundCap)
        pl.setStyle(Qt.PenStyle.CustomDashLine)
        pl.setDashPattern([0.01, 2.5])
        p.setPen(pl)
        p.drawLine(QPointF(172, y), QPointF(214, y))
        p.setPen(T.color(T.APAGADO))
        p.drawText(QPointF(226, y + 5), "Proyección de la letra más probable")


class Vista2D(QWidget):
    pedir_menu = Signal()
    pedir_modo = Signal(str)

    def __init__(self, reconocedor: Reconocedor, parent=None):
        super().__init__(parent)
        self.setObjectName("fondo")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.reconocedor = reconocedor
        self._letra_actual: str | None = None

        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(0, 0, 0, 0)
        raiz.setSpacing(0)
        self.barra = BarraSuperior("2D")
        self.barra.pedir_menu.connect(self.pedir_menu.emit)
        self.barra.pedir_modo.connect(self.pedir_modo.emit)
        raiz.addWidget(self.barra)

        contenido = QHBoxLayout()
        contenido.setContentsMargins(40, 28, 40, 0)
        contenido.setSpacing(32)

        izquierda = Tarjeta()
        caja = QVBoxLayout(izquierda)
        caja.setContentsMargins(24, 22, 24, 24)
        caja.setSpacing(0)
        caja.addWidget(etiqueta("Lienzo 2D", 15, T.TINTA, True))
        caja.addSpacing(4)
        self.subtitulo = etiqueta(SUB_VACIO, 13, T.APAGADO)
        caja.addWidget(self.subtitulo)
        caja.addSpacing(16)
        self.lienzo = Lienzo2D()
        self.lienzo.trazos_cambiaron.connect(self.actualizar)
        caja.addWidget(self.lienzo, 1)
        caja.addSpacing(16)
        self.leyenda = Leyenda()
        self.leyenda.setVisible(False)
        caja.addWidget(self.leyenda)
        contenido.addWidget(izquierda, 1)

        self.panel = PanelClasificacion()
        self.panel.setFixedWidth(468)
        contenido.addWidget(self.panel)
        raiz.addLayout(contenido, 1)

        pie = QHBoxLayout()
        pie.setContentsMargins(40, 24, 40, 30)
        pie.setSpacing(16)
        pie.addStretch(1)
        self.boton_limpiar = boton("Limpiar lienzo", "primario", 190, 46)
        self.boton_limpiar.setToolTip("Borra el trazo")
        self.boton_limpiar.clicked.connect(self.limpiar_lienzo)
        self.boton_reiniciar = boton("Reiniciar captura", "secundario", 190, 46)
        self.boton_reiniciar.setStyleSheet("QPushButton { font-size: 15px; font-weight: bold; }")
        self.boton_reiniciar.setToolTip("Borra el trazo, reinicia el reconocimiento y cierra la guía")
        self.boton_reiniciar.clicked.connect(self.reiniciar_captura)
        pie.addWidget(self.boton_limpiar)
        pie.addWidget(self.boton_reiniciar)
        pie.addStretch(1)
        raiz.addLayout(pie)

        self.capa_guia = CapaGuia(self)
        self.barra.boton_guia.toggled.connect(self._alternar_guia)
        self.capa_guia.cerrada.connect(lambda: self.barra.boton_guia.setChecked(False))
        self.actualizar()

    # ------------------------------------------------------------ acciones
    def limpiar_lienzo(self):
        self.lienzo.limpiar()

    def reiniciar_captura(self):
        self.capa_guia.ocultar()
        self.reconocedor.reiniciar()
        self.lienzo.limpiar()

    def _alternar_guia(self, abrir: bool):
        if abrir:
            b = self.barra.boton_guia
            self.capa_guia.ancla_derecha = b.mapTo(self, b.rect().topRight()).x()
            self.capa_guia.mostrar(self._letra_actual or "a")
        else:
            self.capa_guia.ocultar()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self.capa_guia.isVisible():
            self._alternar_guia(True)

    # ------------------------------------------------------ reconocimiento
    def actualizar(self):
        r = self.reconocedor.procesar(self.lienzo.trazos, dibujando=self.lienzo.dibujando)
        hay_trazo = r.estado != "vacio"
        self.boton_limpiar.setEnabled(hay_trazo)
        self.boton_reiniciar.setEnabled(hay_trazo)
        self.subtitulo.setText(SUB_DIBUJO if hay_trazo else SUB_VACIO)
        self.leyenda.setVisible(r.estado in ("reconocido", "no_definido"))
        self.leyenda.no_definido = r.estado == "no_definido"
        self.leyenda.update()

        if r.estado in ("vacio", "pocos"):
            self._letra_actual = None
            self.panel.mostrar_vacio(dibujando=r.estado == "pocos")
            self.lienzo.mostrar_resultado([], False)
        elif r.estado == "no_definido":
            self._letra_actual = None
            self.panel.mostrar_no_definido(r.probabilidades)
            self.lienzo.mostrar_resultado([], True)
        else:
            self._letra_actual = r.letra
            self.panel.mostrar_ranking(r.probabilidades)
            self.lienzo.mostrar_resultado(r.proyeccion, False)
