"""
PySide6-based GUI for Minerva.
Displays a split-panel window with a chat interface on the left
and a vis.js knowledge graph on the right.
"""
import sys
import os
import json
import threading
import re

# Import heavy AI models before PySide6 to prevent Shiboken import hook slowdowns
from src.chat import Chat

from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtCore import QObject, Slot, Signal, QUrl, QTimer
from src.memory.db import get_session, EntityNode, GraphEdge


class Bridge(QObject):
    """Bridge between the HTML frontend and the Python backend via QWebChannel."""

    # Signal to safely push text to UI from background threads
    response_ready = Signal(str)

    def __init__(self, page, user_name="user"):
        super().__init__()
        self.page = page
        self.user_name = user_name
        self.assistant = Chat(user_name=user_name)

        # Connect signal → slot for thread-safe UI updates
        self.response_ready.connect(self._push_assistant_message)

    @Slot(str)
    def send_message(self, message):
        """Called from JavaScript when the user sends a chat message."""
        print(f"\n[GUI] User message: {message}")

        # Run generation in a background thread to avoid freezing the UI
        thread = threading.Thread(target=self._generate_response, args=(message,), daemon=True)
        thread.start()

    def _generate_response(self, message):
        """Generate the assistant response (runs in a background thread)."""
        full_response = ""
        for token in self.assistant.send_message(message, stream=True):
            full_response += token
            print(token, end="", flush=True)

        print()  # newline after streaming

        # Remove <think>...</think> tags and any extra whitespace created by it
        clean_response = re.sub(r'<think>.*?</think>\s*', '', full_response, flags=re.DOTALL).strip()

        print(f"[GUI] Clean response length: {len(clean_response)} chars")

        # Emit signal to push the response to the UI on the main thread
        self.response_ready.emit(clean_response)

    def _push_assistant_message(self, text):
        """Push the assistant's response to the HTML chat panel (runs on main thread)."""
        escaped = json.dumps(text)
        js = f"appendMessage('assistant', {escaped})"
        self.page.runJavaScript(js)

    def push_graph(self, data):
        """Push graph data (nodes/edges) to the HTML vis.js network."""
        js = f"updateGraph({json.dumps(data)})"
        self.page.runJavaScript(js)


class MainWindow(QMainWindow):
    def __init__(self, user_name="user"):
        super().__init__()
        self.user_name = user_name
        self.setWindowTitle("Minerva — Knowledge Graph")

        self.web = QWebEngineView()

        # Setup QWebChannel BEFORE loading the page so JS can find qt.webChannelTransport
        self.channel = QWebChannel()
        self.bridge = Bridge(self.web.page(), user_name=self.user_name)
        self.channel.registerObject("pyBridge", self.bridge)
        self.web.page().setWebChannel(self.channel)

        # Load HTML using absolute path
        html_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "graph.html"))
        print(f"[GUI] Loading HTML from: {html_path}")
        self.web.load(QUrl.fromLocalFile(html_path))

        # Load graph data once the page finishes loading
        self.web.loadFinished.connect(self.on_load_finished)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.web)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def on_load_finished(self, ok):
        if not ok:
            print("[GUI] ⚠ HTML page failed to load!")
            return

        print("[GUI] ✓ HTML page loaded successfully")

        # Small delay to let JS fully initialize before pushing data
        QTimer.singleShot(300, self.load_graph_data)

        # Polling timer to auto-refresh the graph if DB changes (every 5 seconds)
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.load_graph_data)
        self.poll_timer.start(5000)

    def load_graph_data(self):

        print(f"\n[GUI] Loading graph data for user: '{self.user_name}'")

        with get_session() as session:
            nodes = session.query(EntityNode).filter_by(user_name=self.user_name).all()
            edges = session.query(GraphEdge).filter_by(user_name=self.user_name).all()

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
    app = QApplication(sys.argv)

    window = MainWindow(user_name="user")
    window.resize(1200, 800)
    window.show()

    sys.exit(app.exec())
