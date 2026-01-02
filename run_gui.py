import sys
import os
import multiprocessing
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow
from gui.controllers import MainController


def resource_path(relative_path: str) -> str:
    """Get absolute path to resource, works for dev and for PyInstaller bundle."""
    if hasattr(sys, '_MEIPASS'):
        # Running as PyInstaller bundle
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


def main() -> None:
    """Launch the Qt GUI application."""
    app = QApplication(sys.argv)
    
    icon_path = resource_path('img/favicon.ico')
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    window = MainWindow()
    controller = MainController(window)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    main()