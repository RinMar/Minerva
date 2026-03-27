import os
import sys

# PERFORMANCE_MODE can be "high" or "low"
# "high": All layers on GPU, 40k context, embeddings on GPU
# "low": 24 layers on GPU, 8k context, embeddings on CPU
PERFORMANCE_MODE = "high"
os.environ["MINERVA_PERFORMANCE"] = PERFORMANCE_MODE

# Import GUI and models AFTER setting performance mode environment variable
from src.gui.main import MainWindow  # noqa: E402
from src.memory.db import init_db, get_session, User  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication(sys.argv)
init_db()

# Resolve initial user
from src.config import config
last_user_id = config.get("user", {}).get("last_user_id", 1)

with get_session() as session:
    u = session.query(User).filter_by(id=last_user_id).first()
    if not u:
        # Fallback to first user or create default
        u = session.query(User).first()
        if not u:
            u = User(name="user")
            session.add(u)
            session.commit()
    
    start_id, start_name = u.id, u.name

window = MainWindow(user_id=start_id, user_name=start_name)
window.resize(1400, 850)
window.show()
sys.exit(app.exec())
