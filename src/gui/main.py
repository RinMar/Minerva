"""
PySide6-based GUI for Minerva.
Displays a split-panel window with a chat interface on the left
and a vis.js knowledge graph on the right.
"""
import sys
from src.paths import get_resource_path

from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtCore import Slot, QUrl, QTimer

from src.memory.db import get_session, EntityNode, GraphEdge, User, init_db
from src.gui.bridge import Bridge


class MainWindow(QMainWindow):
    def __init__(self, user_id=1, user_name="user"):
        super().__init__()
        self.user_id = user_id
        self.user_name = user_name
        self.setWindowTitle(f"Minerva — Knowledge Graph ({self.user_name})")

        self.web = QWebEngineView()

        # Setup QWebChannel BEFORE loading the page so JS can find qt.webChannelTransport
        self.channel = QWebChannel()
        self.bridge = Bridge(self.web.page(), user_id=self.user_id, user_name=self.user_name)
        self.channel.registerObject("pyBridge", self.bridge)
        self.web.page().setWebChannel(self.channel)

        # Connect user switch signals
        self.bridge.user_switch_requested.connect(self.on_user_switch_requested)
        self.bridge.profile_changed.connect(self.on_profile_changed)

        # Load HTML using absolute path
        html_path = get_resource_path("resources/gui/index.html")
        print(f"[GUI] Loading HTML from: {html_path}")
        self.web.load(QUrl.fromLocalFile(html_path))

        # Load graph data and start model loading once the page finishes loading
        self.web.loadFinished.connect(self.on_load_finished)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.web)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    @Slot()
    def on_user_switch_requested(self):
        """Deprecated: Handled in HTML/JS."""
        pass

    @Slot(int, str)
    def on_profile_changed(self, user_id, user_name):
        """Update window title and internal state when profile changes in frontend."""
        self.user_id = user_id
        self.user_name = user_name
        self.setWindowTitle(f"Minerva — Knowledge Graph ({self.user_name})")
        print(f"[GUI] Switched to profile: {self.user_name}")
        self.load_graph_data()

    def on_load_finished(self, ok):
        if not ok:
            print("[GUI] WARNING: HTML page failed to load!")
            return

        print("[GUI] HTML page loaded successfully")

        # Start loading AI models in background (loader is already visible in HTML)
        self.bridge.initialize_assistant()

        # Small delay to let JS fully initialize before pushing data
        QTimer.singleShot(300, self.load_graph_data)

        # Polling timer to auto-refresh the graph if DB changes (every 5 seconds)
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.load_graph_data)
        self.poll_timer.start(5000)

    def load_graph_data(self):
        with get_session() as session:
            nodes = session.query(EntityNode).filter_by(user_id=self.user_id).all()
            edges = session.query(GraphEdge).filter_by(user_id=self.user_id).all()

            node_data = []
            edge_data = []

            for node in nodes:
                node_data.append({
                    "id": str(node.id),
                    "label": node.name,
                    "title": node.text or ""
                })

            for edge in edges:
                edge_data.append({
                    "from": str(edge.source_id),
                    "to": str(edge.target_id),
                    "label": edge.relation or ""
                })

        data = {"nodes": node_data, "edges": edge_data}
        self.bridge.push_graph(data)


if __name__ == "__main__":
    init_db()
    app = QApplication(sys.argv)

    # Resolve initial user
    with get_session() as session:
        u = session.query(User).first()
        if not u:
            u = User(name="user")
            session.add(u)
            session.commit()
            start_id, start_name = u.id, u.name
        else:
            start_id, start_name = u.id, u.name

    window = MainWindow(user_id=start_id, user_name=start_name)
    window.resize(1200, 800)
    window.show()

    sys.exit(app.exec())
