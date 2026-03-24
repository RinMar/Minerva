# run.py
"""Entry point for the Minerva GUI application."""
from src.gui.main import MainWindow
from PySide6.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)
window = MainWindow(user_name="user")
window.resize(1400, 850)
window.show()
sys.exit(app.exec())
