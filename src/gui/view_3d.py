"""
view_3d.py
==========
Ventana C: lienzo 3D esférico.

En esta entrega solo funciona el modo 2D: la vista 3D muestra la barra
superior (para volver al menú o al modo 2D) y nada más.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .widgets import BarraSuperior


class Vista3D(QWidget):
    pedir_menu = Signal()
    pedir_modo = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("fondo")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        caja = QVBoxLayout(self)
        caja.setContentsMargins(0, 0, 0, 0)
        caja.setSpacing(0)
        self.barra = BarraSuperior("3D")
        self.barra.boton_guia.hide()          # no hay nada que guiar todavía
        self.barra.pedir_menu.connect(self.pedir_menu.emit)
        self.barra.pedir_modo.connect(self.pedir_modo.emit)
        caja.addWidget(self.barra)
        caja.addStretch(1)                    # contenido vacío a propósito
