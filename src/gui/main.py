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

    stream_start = Signal()
    stream_token = Signal(str)
    stream_state = Signal(str)

    def __init__(self, page, user_name="user"):
        super().__init__()
        self.page = page
        self.user_name = user_name
        self.assistant = Chat(user_name=user_name)

        # Connect signal → slot for thread-safe UI updates
        self.response_ready.connect(self._push_assistant_message)
        self.stream_start.connect(self._push_stream_start)
        self.stream_token.connect(self._push_stream_token)
        self.stream_state.connect(self._push_stream_state)

    @Slot(str)
    def send_message(self, message):
        """Called from JavaScript when the user sends a chat message."""
        print(f"\n[GUI] User message: {message}")

        # Run generation in a background thread to avoid freezing the UI
        thread = threading.Thread(target=self._generate_response, args=(message,), daemon=True)
        thread.start()

    def _parse_buffer_chunk(self, buffer, is_thinking):
        """Parse a chunk of the buffer according to the current state."""
        if is_thinking:
            return self._parse_thinking_chunk(buffer)
        return self._parse_normal_chunk(buffer)

    def _parse_normal_chunk(self, buffer):
        """Parse a chunk when not in thinking mode."""
        # Handle <think> -> enter thinking mode
        if "<think>" in buffer:
            return self._handle_tag(buffer, "<think>", "think_start", True)

        # Handle stray </think> in non-thinking mode (silently consume)
        if "</think>" in buffer:
            return self._handle_tag(buffer, "</think>", None, False)

        # Handle <action:TOOL> start tags
        match = re.search(r'<action:([^>]+)>', buffer)
        if match:
            return self._handle_regex_tag(buffer, match, f"action_start_{match.group(1)}", False)

        # Handle </action:TOOL> end tags
        match_end = re.search(r'</action:([^>]+)>', buffer)
        if match_end:
            return self._handle_regex_tag(buffer, match_end, "action_end", False)

        return self._handle_normal_content(buffer)

    def _handle_tag(self, buffer, tag, state_event, new_is_thinking):
        """Helper to handle exact string tags."""
        pre, post = buffer.split(tag, 1)
        if pre:
            self.stream_token.emit(pre)
        if state_event:
            self.stream_state.emit(state_event)
        return post, new_is_thinking, False

    def _handle_regex_tag(self, buffer, match, state_event, new_is_thinking):
        """Helper to handle regex-based tags."""
        pre = buffer[:match.start()]
        if pre:
            self.stream_token.emit(pre)
        if state_event:
            self.stream_state.emit(state_event)
        return buffer[match.end():], new_is_thinking, False

    def _handle_normal_content(self, buffer):
        """Handle content that contains partial tags or normal text."""
        if "<" not in buffer:
            self.stream_token.emit(buffer)
            return "", False, False

        last_lt = buffer.rfind("<")
        prefix = buffer[last_lt:]

        partial_tags = ["<think>", "</think>", "<action:", "</action:"]
        is_partial = any(
            t.startswith(prefix) and len(prefix) < len(t)
            for t in partial_tags
        )

        if is_partial:
            safe_part = buffer[:last_lt]
            if safe_part:
                self.stream_token.emit(safe_part)
            return prefix, False, True

        self.stream_token.emit(buffer)
        return "", False, False

    def _parse_thinking_chunk(self, buffer):
        """Parse a chunk when in thinking mode."""
        if "</think>" in buffer:
            _, post = buffer.split("</think>", 1)
            self.stream_state.emit("think_end")
            return post, False, False

        if "<" in buffer:
            last_lt = buffer.rfind("<")
            prefix = buffer[last_lt:]
            if "</think>".startswith(prefix) and len(prefix) < len("</think>"):
                return prefix, True, True

        return "", True, False

    def _flush_buffer(self, buffer, is_thinking, strip_leading_whitespace):
        """Flush any remaining content in the buffer."""
        if buffer and not is_thinking:
            if strip_leading_whitespace:
                buffer = buffer.lstrip()
            if buffer:
                self.stream_token.emit(buffer)

    def _generate_response(self, message):
        """Generate the assistant response (runs in a background thread)."""
        self.stream_start.emit()

        is_thinking = False
        strip_leading_whitespace = True
        buffer = ""
        full_response = ""

        for token in self.assistant.send_message(message, stream=True):
            full_response += token
            print(token, end="", flush=True)

            buffer += token

            # State machine to parse <think>...</think> and stream
            while buffer:
                if strip_leading_whitespace and not is_thinking:
                    buffer = buffer.lstrip()
                    if not buffer:
                        break  # wait for more tokens
                    strip_leading_whitespace = False

                buffer, new_is_thinking, break_loop = self._parse_buffer_chunk(buffer, is_thinking)

                if is_thinking and not new_is_thinking:
                    strip_leading_whitespace = True

                is_thinking = new_is_thinking

                if break_loop:
                    break

        self._flush_buffer(buffer, is_thinking, strip_leading_whitespace)

        print()  # newline after streaming
        self.stream_state.emit("done")

        # Cleanup <think> globally for terminal output logs (optional, since UI streamed properly)
        clean_response = re.sub(r'<think>.*?</think>\s*', '', full_response, flags=re.DOTALL).strip()
        print(f"[GUI] Streaming complete. Clean response length: {len(clean_response)} chars")

    def _push_stream_start(self):
        self.page.runJavaScript("startAssistantMessage()")

    def _push_stream_token(self, token):
        escaped = json.dumps(token)
        self.page.runJavaScript(f"appendAssistantToken({escaped})")

    def _push_stream_state(self, state):
        self.page.runJavaScript(f"setAssistantState('{state}')")

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
