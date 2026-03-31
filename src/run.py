import sys
import os
from PySide6.QtWidgets import QApplication

# Fix for PyInstaller windowed mode where sys.stdout/stderr can be None
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

# 1. Initialize the application and show the splash screen as early as possible
app = QApplication(sys.argv)

from src.paths import get_resource_path  # noqa: E402
from src.gui.splash import SplashWidget  # noqa: E402

logo_path = get_resource_path("resources/logo.svg")
splash = SplashWidget(logo_path)
if splash:
    splash.show()
    # Force the OS to process events so the window shows up immediately
    app.processEvents()

# 2. Defer heavy imports until after visual feedback is provided
from src.config import config  # noqa: E402
from src.gui.main import MainWindow  # noqa: E402
from src.memory.db import init_db, get_session, User  # noqa: E402

# 3. Application initialization logic (Database, Profile, etc.)
init_db()

# Resolve initial user
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

# 4. Create and show the main window
window = MainWindow(user_id=start_id, user_name=start_name)
window.resize(1400, 850)
window.show()

# 5. Finish splash screen once the main window is ready
if splash:
    splash.finish(window)

sys.exit(app.exec())
