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
from src.memory.db import get_session, EntityNode, GraphEdge, User, init_db
from src.config import save_last_user


class Bridge(QObject):
    """Bridge between the HTML frontend and the Python backend via QWebChannel."""

    # Signal to safely push text to UI from background threads
    response_ready = Signal(str)

    stream_start = Signal()
    stream_token = Signal(str)
    stream_state = Signal(str)
    user_switch_requested = Signal()
    profile_changed = Signal(int, str)

    def __init__(self, page, user_id, user_name):
        super().__init__()
        self.page = page
        self.user_id = user_id
        self.user_name = user_name
        self.assistant = Chat(user_id=user_id)

        # Connect signal → slot for thread-safe UI updates
        self.response_ready.connect(self._push_assistant_message)
        self.stream_start.connect(self._push_stream_start)
        self.stream_token.connect(self._push_stream_token)
        self.stream_state.connect(self._push_stream_state)

    @Slot()
    def handle_user_click(self):
        """Called from JS when the user button is clicked."""
        print("[GUI] User button clicked")
        # We now handle the dropdown in JS, but we might still want to emit
        # or just let JS handle the display and then call back to other slots.
        self.user_switch_requested.emit()

    @Slot(result=list)
    def get_user_list(self):
        """Return a list of all existing user names."""
        with get_session() as session:
            users = session.query(User).all()
            return [u.name for u in users]

    @Slot(str)
    def switch_profile(self, user_name):
        """Switch to an existing profile by name."""
        with get_session() as session:
            u = session.query(User).filter_by(name=user_name).first()
            if u:
                self.user_id = u.id
                self.user_name = u.name
                save_last_user(self.user_id, self.user_name)
                self.update_user(self.user_id, self.user_name)
                # Notify the window (which owns the bridge) to refresh graph/title
                # Wait, the bridge can't easily notify the window unless we pass the window or use a signal.
                # Actually, MainWindow is connected to user_switch_requested, but that was for QInputDialog.
                # Let's add a signal for the window to react to.
                self.profile_changed.emit(self.user_id, self.user_name)

    @Slot(str)
    def create_profile(self, user_name):
        """Create a new profile and switch to it."""
        if not user_name.strip():
            return

        with get_session() as session:
            existing = session.query(User).filter_by(name=user_name).first()
            if not existing:
                new_u = User(name=user_name)
                session.add(new_u)
                session.commit()
                self.user_id = new_u.id
            else:
                self.user_id = existing.id
            self.user_name = user_name

        save_last_user(self.user_id, self.user_name)
        self.update_user(self.user_id, self.user_name)
        self.profile_changed.emit(self.user_id, self.user_name)

    @Slot(str, str)
    def rename_user(self, old_name, new_name):
        """Rename a user profile in the database."""
        if not new_name.strip() or old_name == new_name:
            return

        with get_session() as session:
            user = session.query(User).filter_by(name=old_name).first()
            if user:
                user.name = new_name
                session.commit()
                print(f"[GUI] Renamed user '{old_name}' to '{new_name}'")
                
                # If the renamed user is the active one, update state
                if self.user_name == old_name:
                    self.user_name = new_name
                    save_last_user(self.user_id, self.user_name)
                    self.profile_changed.emit(self.user_id, self.user_name)

    @Slot()
    def handle_settings_click(self):
        """Called from JS when the settings button is clicked."""
        print("[GUI] Settings button clicked")
        # Placeholder for now
        self.page.runJavaScript("alert('Settings functionality coming soon!')")

    def update_user(self, new_user_id, new_user_name):
        """Update the active user and re-initialize the assistant."""
        print(f"[GUI] Switching bridge to user: '{new_user_name}' (ID: {new_user_id})")
        self.user_id = new_user_id
        self.user_name = new_user_name

        # Reuse existing LLM instance to prevent reloading large weights
        existing_llm = getattr(self.assistant, "llm", None)
        self.assistant = Chat(user_id=new_user_id, llm=existing_llm)
        self.page.runJavaScript("clearChat(); clearGraph();")

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

        print(f"\n[GUI] Loading graph data for user: '{self.user_name}' (ID: {self.user_id})")

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
