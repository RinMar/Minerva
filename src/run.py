from src.gui.main import MainWindow
from src.memory.db import init_db
from PySide6.QtWidgets import QApplication
import os
import sys

# PERFORMANCE_MODE can be "high" or "low"
# "high": All layers on GPU, 40k context, embeddings on GPU
# "low": 24 layers on GPU, 8k context, embeddings on CPU
PERFORMANCE_MODE = "high"
os.environ["MINERVA_PERFORMANCE"] = PERFORMANCE_MODE

# Import GUI and models AFTER setting performance mode environment variable

app = QApplication(sys.argv)
init_db()

window = MainWindow(user_name="user")
window.resize(1400, 850)
window.show()
sys.exit(app.exec())
