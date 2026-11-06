"""
GUI.py — punto de entrada de Glifo.

    python GUI.py

Abre el menú principal. El modo 2D carga los 27 modelos HMM entrenados
(data/modelos_letras_entrenados.json) y reconoce la letra mientras se dibuja.
El modo 3D todavía no tiene contenido.
"""
import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from src.gui import theme
from src.gui.main_window import VentanaPrincipal
from src.gui.recognizer import RUTA_MODELOS, Reconocedor


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Glifo")
    theme.cargar_fuentes()
    app.setStyleSheet(theme.hoja_de_estilos())

    try:
        reconocedor = Reconocedor()
    except (OSError, ValueError, KeyError) as error:
        QMessageBox.critical(
            None, "Glifo",
            f"No se pudieron cargar los modelos HMM desde\n{RUTA_MODELOS}\n\n{error}\n\n"
            "Genera los modelos con: python training/entrenar_hmms.py",
        )
        return 1

    ventana = VentanaPrincipal(reconocedor)
    ventana.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
