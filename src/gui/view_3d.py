"""
view_3d.py
==========
Ventana C: reconocedor y visualizador 3D (lienzo esférico).

Misma estructura que la Ventana B: a la izquierda la esfera con la proyección
punteada; a la derecha el listado ordenado de P(c | O_1:t); abajo, centrados,
«Borrar último trazo», «Borrar todo» y «Centrar vista».

El reconocedor 3D se carga la primera vez que se entra al modo (ver
`VentanaPrincipal.ir_a_modo`), para no alargar el arranque del programa.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from . import theme as T
from .canvas_3d import Lienzo3D
from .guide_popup import CapaGuia
from .ranking_panel import PanelClasificacion
from .view_2d import Leyenda
from .widgets import BarraSuperior, Tarjeta, boton, etiqueta

SUB_VACIO = "Clic izquierdo para dibujar · clic derecho para girar · rueda para acercar"
SUB_DIBUJO = "Cuenta toda la esfera, se vea o no · «arriba» es su polo norte (N), gire como gire"


class Vista3D(QWidget):
    pedir_menu = Signal()
    pedir_modo = Signal(str)

    def __init__(self, reconocedor=None, parent=None):
        super().__init__(parent)
        self.setObjectName("fondo")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.reconocedor = reconocedor
        self._letra_actual: str | None = None

        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(0, 0, 0, 0)
        raiz.setSpacing(0)
        self.barra = BarraSuperior("3D")
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
        caja.addWidget(etiqueta("Esfera 3D", 15, T.TINTA, True))
        caja.addSpacing(4)
        self.subtitulo = etiqueta(SUB_VACIO, 13, T.APAGADO)
        caja.addWidget(self.subtitulo)
        caja.addSpacing(16)
        self.lienzo = Lienzo3D()
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
        self.boton_limpiar = boton("Borrar último trazo", "primario", 190, 46)
        self.boton_limpiar.setToolTip("Quita solo el último trazo que dibujaste")
        self.boton_limpiar.clicked.connect(self.borrar_ultimo)
        self.boton_reiniciar = boton("Borrar todo", "secundario", 190, 46)
        self.boton_reiniciar.setStyleSheet("QPushButton { font-size: 15px; font-weight: bold; }")
        self.boton_reiniciar.setToolTip("Borra todos los trazos y empieza de nuevo (la esfera no se mueve)")
        self.boton_reiniciar.clicked.connect(self.borrar_todo)
        self.boton_centrar = boton("Centrar vista", "secundario", 190, 46)
        self.boton_centrar.setStyleSheet("QPushButton { font-size: 15px; font-weight: bold; }")
        self.boton_centrar.setToolTip("Vuelve a la orientación y al tamaño iniciales de la esfera, sin borrar")
        self.boton_centrar.clicked.connect(self.lienzo.centrar_vista)
        for b in (self.boton_limpiar, self.boton_reiniciar, self.boton_centrar):
            pie.addWidget(b)
        pie.addStretch(1)
        raiz.addLayout(pie)

        self.capa_guia = CapaGuia(self)
        self.barra.boton_guia.toggled.connect(self._alternar_guia)
        self.capa_guia.cerrada.connect(lambda: self.barra.boton_guia.setChecked(False))
        self.actualizar()

    # ------------------------------------------------------------ acciones
    def fijar_reconocedor(self, reconocedor) -> None:
        self.reconocedor = reconocedor
        self.actualizar()

    def borrar_ultimo(self):
        self.lienzo.borrar_ultimo()

    def borrar_todo(self):
        if self.reconocedor is not None:
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
        if self.reconocedor is None:
            r_estado = "vacio" if self.lienzo.vacio else "pocos"
            self._pintar_vacio(r_estado)
            return
        r = self.reconocedor.procesar_3d(self.lienzo.trazos, dibujando=self.lienzo.dibujando,
                                         zoom=self.lienzo.zoom)
        hay_trazo = not self.lienzo.vacio
        self.boton_limpiar.setEnabled(hay_trazo)
        self.boton_reiniciar.setEnabled(hay_trazo)
        self.subtitulo.setText(SUB_DIBUJO if hay_trazo else SUB_VACIO)
        self.leyenda.setVisible(r.estado in ("reconocido", "no_definido"))
        self.leyenda.no_definido = r.estado == "no_definido"
        self.leyenda.update()

        if r.estado in ("vacio", "pocos"):
            self._pintar_vacio(r.estado)
        elif r.estado == "no_definido":
            self._letra_actual = None
            self.panel.mostrar_no_definido(r.probabilidades)
            self.lienzo.mostrar_resultado([], True)
        else:
            self._letra_actual = r.letra
            self.panel.mostrar_ranking(r.probabilidades)
            self.lienzo.mostrar_resultado(r.proyeccion, False)

    def _pintar_vacio(self, estado: str):
        self._letra_actual = None
        self.panel.mostrar_vacio(dibujando=estado == "pocos")
        self.lienzo.mostrar_resultado([], False)
        hay_trazo = not self.lienzo.vacio
        self.boton_limpiar.setEnabled(hay_trazo)
        self.boton_reiniciar.setEnabled(hay_trazo)
        self.leyenda.setVisible(False)
