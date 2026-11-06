"""
theme.py
========
Colores, tipografías y estilos compartidos por todas las vistas.

Los valores salen de los mockups y de Glifo Capture, para que la aplicación
se vea igual que la página de captura del dataset:

* Texto: Selawik (reemplazo libre de Segoe UI, Microsoft, OFL).
* La «a» y la «q»: Playwrite AR (letra escolar; la «q» lleva una cruz abajo).
* Números: Inconsolata.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase

RAIZ = Path(__file__).resolve().parents[2]
CARPETA_FUENTES = RAIZ / "assets" / "fonts"

# --------------------------------------------------------------------------
# Colores (mismos tokens que los mockups)
# --------------------------------------------------------------------------
FONDO = "#f2f6f9"
SUPERFICIE = "#ffffff"
BORDE = "#dde7ee"
CHIP = "#e4eef5"
TINTA = "#1b4560"
TEXTO = "#2f4656"
APAGADO = "#6c8191"
PRIMARIO = "#2b6a94"
PRIMARIO_SUAVE = "#e8f2f9"
TEAL = "#3d8f7c"
TEAL_TRAZO = "#5aa596"
TEAL_SUAVE = "#dcefe9"
GUIA = "#c9d8e3"
GUIA_FUERTE = "#9fbbcf"
GRIS_LETRA = "#8aa0b0"
AVISO_FONDO = "#fdf6ea"
AVISO_BORDE = "#f0d9a8"
AVISO_TEXTO = "#6b4610"
AVISO_ICONO = "#a8701a"
ROJO = "#d64545"
ROJO_OSCURO = "#b83232"

# Líneas guía, en fracción del lado del lienzo (las mismas de Glifo Capture:
# 180, 420, 700 y 900 sobre 1000).
LINEAS_GUIA = (
    ("ascendente", 0.18),
    ("altura x", 0.42),
    ("línea base", 0.70),
    ("descendente", 0.90),
)

LETRAS = list("abcdefghijklmnñopqrstuvwxyz")

# --------------------------------------------------------------------------
# Tipografías
# --------------------------------------------------------------------------
_familias = {"texto": "Segoe UI", "escolar": "Segoe UI", "numeros": "Consolas", "numeros_fuerte": "Consolas"}


def cargar_fuentes() -> None:
    """Registra las fuentes de assets/fonts. Llamar una vez, con la QApplication creada."""
    archivos = {
        "texto": ["Selawik-Regular.ttf", "Selawik-Bold.ttf"],
        "escolar": ["PlaywriteAR-aq.ttf"],
        "numeros": ["Inconsolata-Regular.ttf"],
        "numeros_fuerte": ["Inconsolata-SemiBold.ttf"],
    }
    for rol, nombres in archivos.items():
        for nombre in nombres:
            ruta = CARPETA_FUENTES / nombre
            if not ruta.exists():
                continue
            ident = QFontDatabase.addApplicationFont(str(ruta))
            familias = QFontDatabase.applicationFontFamilies(ident) if ident >= 0 else []
            if familias:
                _familias[rol] = familias[0]


def familia(rol: str) -> str:
    return _familias[rol]


def fuente(tam: float, negrita: bool = False, rol: str = "texto") -> QFont:
    f = QFont(_familias[rol])
    f.setPixelSize(round(tam))
    f.setWeight(QFont.Weight.Bold if negrita else QFont.Weight.Normal)
    f.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    return f


def fuente_letra(letra: str, tam: float) -> QFont:
    """Fuente para mostrar una letra del alfabeto: la «a» y la «q» van en letra escolar."""
    if letra in ("a", "q"):
        # Playwrite AR tiene el ojo más pequeño: se agranda un poco para igualar la altura x.
        return fuente(tam * 1.04, rol="escolar")
    return fuente(tam)


def html_letra(letra: str) -> str:
    """La letra como HTML, con la letra escolar si es «a» o «q» (para textos mixtos)."""
    if letra in ("a", "q"):
        return f'<span style="font-family:\'{_familias["escolar"]}\'">{letra}</span>'
    return letra


def color(hexa: str, alfa: float = 1.0) -> QColor:
    c = QColor(hexa)
    c.setAlphaF(alfa)
    return c


# --------------------------------------------------------------------------
# Hoja de estilos base (Qt Style Sheets)
# --------------------------------------------------------------------------
def hoja_de_estilos() -> str:
    t = _familias["texto"]
    return f"""
    QWidget {{ font-family: '{t}'; color: {TEXTO}; }}
    QMainWindow, QStackedWidget, #fondo {{ background: {FONDO}; }}
    #tarjeta {{ background: {SUPERFICIE}; border: 1px solid {BORDE}; border-radius: 16px; }}
    #barra {{ background: {SUPERFICIE}; border-bottom: 1px solid {BORDE}; }}
    QPushButton {{ font-family: '{t}'; }}
    QPushButton#secundario {{
        background: {SUPERFICIE}; color: {PRIMARIO}; border: 1px solid {BORDE};
        border-radius: 10px; padding: 0 22px; font-size: 14px;
    }}
    QPushButton#secundario:hover {{ border-color: {PRIMARIO}; }}
    QPushButton#secundario:disabled {{ background: #e6edf2; color: #9eb0bd; border-color: #e6edf2; }}
    QPushButton#primario {{
        background: {PRIMARIO}; color: white; border: 1px solid {PRIMARIO};
        border-radius: 10px; padding: 0 22px; font-size: 15px; font-weight: bold;
    }}
    QPushButton#primario:hover {{ background: #24597d; }}
    QPushButton#primario:disabled {{ background: #e6edf2; color: #9eb0bd; border-color: #e6edf2; }}
    QPushButton#guia {{
        background: {SUPERFICIE}; color: {PRIMARIO}; border: 1px solid {BORDE};
        border-radius: 10px; padding: 0 16px; font-size: 14px; font-weight: bold;
    }}
    QPushButton#guia:checked {{ background: #eaf3f9; border: 1.5px solid {PRIMARIO}; }}
    QPushButton#guia:hover {{ border-color: {PRIMARIO}; }}
    #conmutador {{ background: {CHIP}; border-radius: 12px; }}
    QPushButton#modo {{
        background: transparent; color: {PRIMARIO}; border: none; border-radius: 8px;
        padding: 0 22px; font-size: 14px; font-weight: bold;
    }}
    QPushButton#modo:checked {{ background: {PRIMARIO}; color: white; }}
    QPushButton#salir {{
        background: {ROJO}; color: white; border: 1px solid {ROJO}; border-radius: 10px;
    }}
    QPushButton#salir:hover {{ background: {ROJO_OSCURO}; border-color: {ROJO_OSCURO}; }}
    QToolTip {{ background: {TINTA}; color: white; border: none; padding: 4px 8px; }}
    """
