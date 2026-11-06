"""
main_window.py
==============
Ventana principal: funciona como las secciones del HTML de los mockups.
Solo una sección es visible a la vez (menú, 2D o 3D) y se cambia entre ellas
con los botones de cada vista.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QStackedWidget

from .menu_view import VistaMenu
from .recognizer import Reconocedor
from .view_2d import Vista2D
from .view_3d import Vista3D


class VentanaPrincipal(QMainWindow):
    def __init__(self, reconocedor: Reconocedor):
        super().__init__()
        self.setWindowTitle("Glifo")
        self.resize(1440, 900)
        self.setMinimumSize(1200, 800)

        self.secciones = QStackedWidget()
        self.setCentralWidget(self.secciones)

        self.menu = VistaMenu()
        self.vista_2d = Vista2D(reconocedor)
        self.vista_3d = Vista3D()
        for vista in (self.menu, self.vista_2d, self.vista_3d):
            self.secciones.addWidget(vista)

        self.menu.elegir_modo.connect(self.ir_a_modo)
        self.menu.salir.connect(self.close)
        for vista in (self.vista_2d, self.vista_3d):
            vista.pedir_menu.connect(self.ir_al_menu)
            vista.pedir_modo.connect(self.ir_a_modo)
        self.ir_al_menu()

    def ir_al_menu(self):
        self.vista_2d.capa_guia.ocultar()
        self.secciones.setCurrentWidget(self.menu)

    def ir_a_modo(self, modo: str):
        self.vista_2d.capa_guia.ocultar()
        vista = self.vista_2d if modo == "2D" else self.vista_3d
        vista.barra.fijar_modo(modo)
        self.secciones.setCurrentWidget(vista)

    def keyPressEvent(self, ev):
        # Atajo: Esc vuelve al menú desde cualquier modo.
        if ev.key() == Qt.Key.Key_Escape and self.secciones.currentWidget() is not self.menu:
            self.ir_al_menu()
        else:
            super().keyPressEvent(ev)
