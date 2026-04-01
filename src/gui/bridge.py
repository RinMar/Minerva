"""
Bridge between the HTML frontend and the Python backend via QWebChannel.
"""
import json
import threading
import re
from PySide6.QtCore import QObject, Slot, Signal

from src.config import config, update_config_mode, save_performance_mode, save_last_user
from src.memory.db import get_session, User
from src.gui.stream_parser import StreamParser


class Bridge(QObject):
    """Bridge between the HTML frontend and the Python backend via QWebChannel."""

    # Signal to safely push text to UI from background threads
    response_ready = Signal(str)

    stream_start = Signal()
    stream_token = Signal(str)
    stream_state = Signal(str)
    user_switch_requested = Signal()
    profile_changed = Signal(int, str)
    model_loading_status = Signal(bool, str)  # (is_loading, message)

    def __init__(self, page, user_id, user_name):
        super().__init__()
        self.page = page
        self.user_id = user_id
        self.user_name = user_name
        self.assistant = None  # Initialized in background

        # Connect signal → slot for thread-safe UI updates
        self.response_ready.connect(self._push_assistant_message)
        self.stream_start.connect(self._push_stream_start)
        self.stream_token.connect(self._push_stream_token)
        self.stream_state.connect(self._push_stream_state)
        self.model_loading_status.connect(self._handle_model_loading_status)

    def _handle_model_loading_status(self, is_loading, message):
        """Thread-safe slot to toggle the UI loader."""
        if is_loading:
            escaped_msg = json.dumps(message)
            self.page.runJavaScript(f"showModelLoader({escaped_msg})")
        else:
            self.page.runJavaScript("hideModelLoader()")

    def initialize_assistant(self, existing_llm=None):
        """Start the assistant initialization in a background thread."""
        thread = threading.Thread(
            target=self._bg_init_assistant,
            args=(existing_llm,),
            daemon=True
        )
        thread.start()

    def _bg_init_assistant(self, existing_llm=None):
        """Perform the actual model loading in the background."""
        msg = "Loading AI Models..." if not existing_llm else "Switching Profile..."
        self.model_loading_status.emit(True, msg)

        try:
            # 1. PEAK PERFORMANCE: We defer the heavy imports to the actual background thread
            # so the main UI thread never touches torch or llama-cpp during import/MainWindow init.
            from src.chat import Chat

            # Initialize the Chat object (this triggers heavy model loading/warm-up)
            self.assistant = Chat(user_id=self.user_id, llm=existing_llm)

            # Ensure embedding/reranker are also warmed up while we show the loader
            # if they haven't been loaded yet.
            if hasattr(self.assistant, "emb_model"):
                _ = self.assistant.emb_model
            if hasattr(self.assistant, "cross_encoder"):
                _ = self.assistant.cross_encoder

            print(f"[GUI] Assistant initialized for user {self.user_id}")
        except Exception as e:
            try:
                print(f"[GUI] WARNING: Error during model loading: {e}")
            except Exception:
                pass  # Swallow print encoding errors on Windows
        finally:
            # ALWAYS hide the loader, even if initialization failed
            self.model_loading_status.emit(False, "")

    @Slot()
    def handle_user_click(self):
        """Called from JS when the user button is clicked."""
        print("[GUI] User button clicked")
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

                if self.user_name == old_name:
                    self.user_name = new_name
                    save_last_user(self.user_id, self.user_name)
                    self.profile_changed.emit(self.user_id, self.user_name)

    @Slot()
    def handle_settings_click(self):
        """Called from JS when the settings button is clicked."""
        print("[GUI] Settings button clicked")
        self.page.runJavaScript("showSettings()")

    def update_user(self, new_user_id, new_user_name):
        """Update the active user and re-initialize the assistant."""
        print(f"[GUI] Switching bridge to user: '{new_user_name}' (ID: {new_user_id})")
        self.user_id = new_user_id
        self.user_name = new_user_name

        # Reuse existing LLM instance to prevent reloading large weights
        existing_llm = getattr(self.assistant, "llm", None) if self.assistant else None
        self.page.runJavaScript("clearChat(); clearGraph();")
        self.initialize_assistant(existing_llm=existing_llm)

    @Slot(str)
    def update_performance_mode(self, mode):
        """Switch performance mode, reload config, and re-initialize models."""
        print(f"[GUI] Requested performance mode switch: {mode}")
        save_performance_mode(mode)
        update_config_mode(mode)
        from src.models.embeddings import reset_models
        reset_models()
        self.page.runJavaScript("clearChat();")
        self.initialize_assistant(existing_llm=None)

    @Slot(result=str)
    def get_performance_mode(self):
        """Return the current performance mode for the UI."""
        return config.get("performance_mode", "low")

    @Slot(str)
    def send_message(self, message):
        """Called from JavaScript when the user sends a chat message."""
        if not self.assistant:
            print("[GUI] WARNING: Cannot send message: Assistant not initialized")
            return

        print(f"\n[GUI] User message: {message}")

        # Run generation in a background thread to avoid freezing the UI
        thread = threading.Thread(target=self._generate_response, args=(message,), daemon=True)
        thread.start()

    def _generate_response(self, message):
        """Generate the assistant response (runs in a background thread)."""
        self.stream_start.emit()
        full_response = ""

        parser = StreamParser(
            on_token=self.stream_token.emit,
            on_state=self.stream_state.emit
        )

        for token in self.assistant.send_message(message, stream=True):
            full_response += token
            print(token, end="", flush=True)
            parser.process_token(token)

        parser.flush()

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
