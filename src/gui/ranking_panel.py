"""
ranking_panel.py
================
Panel derecho: listado ordenado de mayor a menor de P(c | O_1:t), con la letra
como ícono de referencia, su barra y su porcentaje.

Estados: vacío (las 27 letras), «sigue dibujando», ranking y trazo no definido.
"""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from . import theme as T
from .widgets import FichaLetra, IconoAviso, Logo, Tarjeta, etiqueta

FILAS_VISIBLES = 4


def formato_porcentaje(p: float) -> str:
    if 0 < p < 0.005:
        return "<1 %"
    return f"{round(p * 100)} %"


class Barra(QWidget):
    def __init__(self, ancho: int = 230, alto: int = 8, parent=None):
        super().__init__(parent)
        self.setFixedSize(ancho, alto)
        self.valor = 0.0
        self.color = T.TEAL_TRAZO

    def fijar(self, valor: float, color: str):
        self.valor = max(0.0, min(1.0, valor))
        self.color = color
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        r = self.height() / 2
        p.setBrush(T.color("#e6eef3"))
        p.drawRoundedRect(QRectF(self.rect()), r, r)
        if self.valor > 0:
            ancho = max(self.height(), self.width() * self.valor)
            p.setBrush(T.color(self.color))
            p.drawRoundedRect(QRectF(0, 0, ancho, self.height()), r, r)


class FilaLetra(QFrame):
    """Una fila del ranking: ficha de la letra, nombre, barra y porcentaje."""

    def __init__(self, compacta: bool = False, parent=None):
        super().__init__(parent)
        self.compacta = compacta
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(58 if compacta else 80)
        fila = QHBoxLayout(self)
        fila.setContentsMargins(16 if compacta else 18, 0, 18, 0)
        fila.setSpacing(16)
        lado = 40 if compacta else 56
        self.ficha = FichaLetra("a", lado, 22 if compacta else 30, T.TEAL_SUAVE, T.TINTA, 10)
        fila.addWidget(self.ficha)
        centro = QVBoxLayout()
        centro.setSpacing(8)
        centro.addStretch(1)
        self.nombre = QLabel()
        self.nombre.setTextFormat(Qt.TextFormat.RichText)
        self.nombre.setFont(T.fuente(14))
        centro.addWidget(self.nombre)
        self.barra = Barra(230, 8)
        centro.addWidget(self.barra)
        centro.addStretch(1)
        fila.addLayout(centro)
        fila.addStretch(1)
        self.porcentaje = QLabel()
        self.porcentaje.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.porcentaje.setFont(T.fuente(16 if compacta else 19, True))
        fila.addWidget(self.porcentaje)

    def mostrar(self, letra: str, prob: float, destacada: bool = False, otras: int | None = None):
        if otras is not None:
            self.ficha.configurar(letra=f"+{otras}", fondo="#e2eaef", color_letra=T.APAGADO)
            self.ficha._tam = 15 if self.compacta else 19
            nombre_html = f"Otras {otras} letras"
            color_nombre, color_barra = T.APAGADO, "#9fb3c1"
            fondo, borde = "#f2f6f9", "transparent"
        else:
            self.ficha._tam = 22 if self.compacta else 30
            if self.compacta:
                self.ficha.configurar(letra=letra, fondo="#eef3f6", color_letra=T.APAGADO)
            else:
                self.ficha.configurar(letra=letra, fondo=T.TEAL_SUAVE, color_letra=T.TINTA)
            nombre_html = f"Letra «{T.html_letra(letra)}»"
            color_nombre = "#59707f" if self.compacta else T.TINTA
            color_barra = "#b9c8d2" if self.compacta else (T.PRIMARIO if destacada else T.TEAL_TRAZO)
            fondo = T.PRIMARIO_SUAVE if destacada else T.SUPERFICIE
            borde = T.PRIMARIO if destacada else T.BORDE
        peso = "bold" if destacada else "normal"
        self.nombre.setText(f'<span style="color:{color_nombre}; font-weight:{peso}">{nombre_html}</span>')
        self.barra.fijar(prob, color_barra)
        color_pct = T.PRIMARIO if destacada else ("#8aa0b0" if self.compacta else "#59707f")
        self.porcentaje.setText(formato_porcentaje(prob))
        self.porcentaje.setStyleSheet(f"color: {color_pct}; background: transparent;")
        ancho_borde = "1.5px" if destacada else "1px"
        self.setStyleSheet(
            f"FilaLetra {{ background: {fondo}; border: {ancho_borde} solid {borde}; border-radius: 12px; }}"
        )


class PanelClasificacion(Tarjeta):
    def __init__(self, parent=None):
        super().__init__(parent)
        caja = QVBoxLayout(self)
        caja.setContentsMargins(24, 24, 24, 24)
        caja.setSpacing(0)
        caja.addWidget(etiqueta("Clasificación en tiempo real", 20, T.TINTA, True))
        caja.addSpacing(4)
        self.subtitulo = etiqueta("Aquí aparecerán las letras más probables", 13, T.APAGADO)
        caja.addWidget(self.subtitulo)
        caja.addSpacing(20)

        self.paginas = QStackedWidget()
        self.paginas.setStyleSheet("background: transparent;")
        caja.addWidget(self.paginas, 1)
        self._pagina_vacia()
        self._pagina_ranking()
        self._pagina_no_definido()
        self.mostrar_vacio()

    # ----------------------------------------------------------- páginas
    def _pagina_vacia(self):
        pag = QWidget()
        caja = QVBoxLayout(pag)
        caja.setContentsMargins(0, 0, 0, 0)
        caja.setSpacing(0)
        aviso = QFrame()
        aviso.setObjectName("avisoVacio")
        aviso.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        aviso.setStyleSheet("#avisoVacio { background: #e3eff7; border-radius: 12px; }")
        fila = QHBoxLayout(aviso)
        fila.setContentsMargins(22, 20, 22, 20)
        fila.setSpacing(22)
        fila.addWidget(Logo(56), 0, Qt.AlignmentFlag.AlignVCenter)
        textos = QVBoxLayout()
        textos.setSpacing(6)
        self.vacio_titulo = etiqueta("Aún no hay trazo", 16, T.TINTA, True)
        self.vacio_texto = etiqueta("Dibuja una letra en el lienzo y aquí verás\nsus probabilidades, de mayor a menor.",
                                    13, T.APAGADO)
        textos.addWidget(self.vacio_titulo)
        textos.addWidget(self.vacio_texto)
        fila.addLayout(textos, 1)
        caja.addWidget(aviso)
        caja.addSpacing(22)
        caja.addWidget(etiqueta("Las 27 letras que reconoce Glifo", 13, T.APAGADO, True))
        caja.addSpacing(10)
        rejilla = QGridLayout()
        rejilla.setHorizontalSpacing(8)
        rejilla.setVerticalSpacing(8)
        for i, letra in enumerate(T.LETRAS):
            rejilla.addWidget(FichaLetra(letra), i // 6, i % 6)
        caja.addLayout(rejilla)
        caja.addStretch(1)
        self.paginas.addWidget(pag)

    def _pagina_ranking(self):
        pag = QWidget()
        caja = QVBoxLayout(pag)
        caja.setContentsMargins(0, 0, 0, 0)
        caja.setSpacing(12)
        self.filas = [FilaLetra() for _ in range(FILAS_VISIBLES)]
        for f in self.filas:
            caja.addWidget(f)
        self.fila_otras = FilaLetra()
        caja.addWidget(self.fila_otras)
        caja.addStretch(1)
        self.paginas.addWidget(pag)

    def _pagina_no_definido(self):
        pag = QWidget()
        caja = QVBoxLayout(pag)
        caja.setContentsMargins(0, 0, 0, 0)
        caja.setSpacing(0)
        aviso = QFrame()
        aviso.setObjectName("avisoNoDef")
        aviso.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        aviso.setStyleSheet(f"#avisoNoDef {{ background: {T.AVISO_FONDO}; border: 1px solid {T.AVISO_BORDE};"
                            " border-radius: 12px; }")
        fila = QHBoxLayout(aviso)
        fila.setContentsMargins(18, 16, 18, 16)
        fila.setSpacing(14)
        circulo = QFrame()
        circulo.setFixedSize(40, 40)
        circulo.setStyleSheet("background: #f6e3bd; border-radius: 20px;")
        dentro = QHBoxLayout(circulo)
        dentro.setContentsMargins(9, 8, 9, 10)
        dentro.addWidget(IconoAviso(22))
        fila.addWidget(circulo, 0, Qt.AlignmentFlag.AlignTop)
        textos = QVBoxLayout()
        textos.setSpacing(4)
        textos.addWidget(etiqueta("Trazo no definido", 16, T.AVISO_TEXTO, True))
        explicacion = etiqueta("El trazo no se parece a ninguna de las 27 letras: hasta la más probable "
                               "lo explica peor que casi todas las letras de entrenamiento.", 13, "#7a5a2a")
        explicacion.setWordWrap(True)
        textos.addWidget(explicacion)
        textos.addSpacing(4)
        consejos = QLabel("• Dibuja una sola letra minúscula.<br>• Usa las líneas guía para el tamaño.<br>"
                          "• Si tienes dudas, abre la Guía de letras.")
        consejos.setFont(T.fuente(13))
        consejos.setStyleSheet("color: #5f4a2a; background: transparent;")
        textos.addWidget(consejos)
        fila.addLayout(textos, 1)
        caja.addWidget(aviso)
        caja.addSpacing(18)
        caja.addWidget(etiqueta("Probabilidades actuales del modelo", 13, T.APAGADO, True))
        caja.addSpacing(8)
        self.filas_mini = [FilaLetra(compacta=True) for _ in range(FILAS_VISIBLES)]
        for f in self.filas_mini:
            caja.addWidget(f)
            caja.addSpacing(8)
        self.fila_otras_mini = FilaLetra(compacta=True)
        caja.addWidget(self.fila_otras_mini)
        caja.addStretch(1)
        self.paginas.addWidget(pag)

    # ----------------------------------------------------------- estados
    def mostrar_vacio(self, dibujando: bool = False):
        if dibujando:
            self.vacio_titulo.setText("Sigue dibujando")
            self.vacio_texto.setText("Con un poco más de trazo aparecerán\nlas letras más probables.")
        else:
            self.vacio_titulo.setText("Aún no hay trazo")
            self.vacio_texto.setText("Dibuja una letra en el lienzo y aquí verás\nsus probabilidades, de mayor a menor.")
        self.subtitulo.setText("Aquí aparecerán las letras más probables")
        self.paginas.setCurrentIndex(0)

    def _llenar(self, filas, fila_otras, probabilidades, destacar: bool):
        for i, fila in enumerate(filas):
            letra, p = probabilidades[i]
            fila.mostrar(letra, p, destacada=destacar and i == 0)
        resto = sum(p for _, p in probabilidades[len(filas):])
        fila_otras.mostrar("", resto, otras=len(probabilidades) - len(filas))

    def mostrar_ranking(self, probabilidades):
        self.subtitulo.setText("Probabilidad de cada letra según el trazo actual")
        self._llenar(self.filas, self.fila_otras, probabilidades, destacar=True)
        self.paginas.setCurrentIndex(1)

    def mostrar_no_definido(self, probabilidades):
        self.subtitulo.setText("Probabilidad de cada letra según el trazo actual")
        self._llenar(self.filas_mini, self.fila_otras_mini, probabilidades, destacar=False)
        self.paginas.setCurrentIndex(2)
