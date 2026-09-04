"""
Bridge between the HTML frontend and the Python backend via QWebChannel.
"""
import json
import threading
import re
from PySide6.QtCore import QObject, Slot, Signal

from src.config import (
    config, update_model_settings, save_last_user,
    MODEL_TOTAL_LAYERS, MODEL_PER_LAYER_MB, KV_PER_TOKEN_PER_LAYER_MB,
    VRAM_FIXED_OVERHEAD_MB, CTX_MIN, CTX_MAX
)
from src.memory.db import get_session, User
from src.gui.stream_parser import StreamParser
from src.utils.vram import (
    get_initial_vram_info,
    estimate_max_gpu_layers,
    estimate_max_context,
)


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
    model_warning = Signal(str)  # Warning message banner for UI

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
        self.model_warning.connect(self._handle_model_warning)

    def _handle_model_loading_status(self, is_loading, message):
        """Thread-safe slot to toggle the UI loader."""
        if is_loading:
            escaped_msg = json.dumps(message)
            self.page.runJavaScript(f"showModelLoader({escaped_msg})")
        else:
            self.page.runJavaScript("hideModelLoader()")

    def _handle_model_warning(self, message):
        """Thread-safe slot to display a model warning banner in the chat UI."""
        escaped_msg = json.dumps(message)
        self.page.runJavaScript(f"showModelWarning({escaped_msg})")

    def initialize_assistant(self, existing_llm=None):
        """Start the assistant initialization in a background thread."""
        thread = threading.Thread(
            target=self._bg_init_assistant,
            args=(existing_llm,),
            daemon=True
        )
        thread.start()

    def _bg_init_assistant(self, existing_llm=None):
        """Perform the actual model loading in the background with proactive VRAM check & OOM fallback."""
        msg = "Loading AI Models..." if not existing_llm else "Switching Profile..."
        self.model_loading_status.emit(True, msg)

        try:
            from src.chat import Chat
            from src.models.base_llm import CustomLLM

            llm_to_use = existing_llm

            if llm_to_use is None:
                # Proactive VRAM check to avoid OOM
                from src.utils.vram import get_initial_vram_info
                vram_info = get_initial_vram_info()
                free_vram = vram_info.get("free_mb", 0.0)
                req_layers = config.get("llm", {}).get("n_gpu_layers", 0)
                req_ctx = config.get("llm", {}).get("n_ctx", 8192)

                if free_vram > 0 and req_layers > 0:
                    safe_layers = estimate_max_gpu_layers(
                        free_vram_mb=free_vram,
                        n_ctx=req_ctx,
                        total_layers=MODEL_TOTAL_LAYERS,
                    )

                    if req_layers > safe_layers:
                        safe_ctx = estimate_max_context(
                            free_vram_mb=free_vram,
                            n_gpu_layers=safe_layers,
                            ctx_min=CTX_MIN,
                            ctx_max=req_ctx,
                        )
                        print(
                            f"[GUI] Proactive VRAM adjustment: Requested {req_layers} layers, "
                            f"clamping to {safe_layers}/{MODEL_TOTAL_LAYERS} layers ({safe_ctx} ctx) "
                            f"for {free_vram:.0f}MB free VRAM."
                        )
                        update_model_settings(safe_layers, safe_ctx)

                        warn_text = (
                            f"⚠️ Insufficient VRAM for requested config. "
                            f"Auto-configured to {safe_layers}/{MODEL_TOTAL_LAYERS} GPU layers, "
                            f"{safe_ctx:,} context tokens."
                        )
                        self.model_warning.emit(warn_text)

            # Attempt model load
            try:
                self.assistant = Chat(user_id=self.user_id, llm=llm_to_use)
            except Exception as load_err:
                print(f"[GUI] WARNING: Primary model loading failed: {load_err}")

                # Reactive Fallback logic as secondary safety net
                vram_info = get_initial_vram_info()
                current_ctx = config.get("llm", {}).get("n_ctx", 8192)
                req_layers = config.get("llm", {}).get("n_gpu_layers", 0)
                free_vram = vram_info.get("free_mb", 0.0)

                safe_layers = estimate_max_gpu_layers(
                    free_vram_mb=free_vram,
                    n_ctx=current_ctx,
                    total_layers=MODEL_TOTAL_LAYERS,
                )
                if safe_layers >= req_layers:
                    safe_layers = max(0, safe_layers - 10)

                print(
                    f"[GUI] Attempting fallback load with reduced GPU layers: "
                    f"{safe_layers}/{MODEL_TOTAL_LAYERS}..."
                )
                self.model_loading_status.emit(True, f"Retrying with {safe_layers} GPU layers...")

                fallback_llm = CustomLLM(
                    n_gpu_layers=safe_layers,
                    n_ctx=current_ctx,
                )
                self.assistant = Chat(user_id=self.user_id, llm=fallback_llm)
                update_model_settings(safe_layers, current_ctx)

                warn_text = (
                    f"⚠️ Insufficient GPU VRAM for model loading. "
                    f"Auto-offloaded {safe_layers}/{MODEL_TOTAL_LAYERS} layers to GPU."
                )
                self.model_warning.emit(warn_text)

            # Ensure embedding/reranker models are warmed up
            if hasattr(self.assistant, "emb_model"):
                _ = self.assistant.emb_model
            if hasattr(self.assistant, "cross_encoder"):
                _ = self.assistant.cross_encoder

            print(f"[GUI] Assistant successfully initialized for user {self.user_id}")

        except Exception as final_err:
            print(f"[GUI] CRITICAL: Failed to load models after fallback: {final_err}")
            self.assistant = None
            self.model_warning.emit("❌ Failed to load AI models. Please reduce GPU layers in settings.")
        finally:
            # ALWAYS hide the loader overlay
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

    @Slot(int, int)
    def update_model_settings(self, n_gpu_layers, n_ctx):
        """Update model GPU layers and context size, reload config, and re-initialize models."""
        print(f"[GUI] Requested model settings update: n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx}")
        update_model_settings(n_gpu_layers, n_ctx)
        from src.models.embeddings import reset_models
        reset_models()
        self.page.runJavaScript("clearChat(); hideModelWarning();")

        # Explicitly free the old assistant before reloading to ensure VRAM is released
        if self.assistant:
            self.assistant = None
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        self.initialize_assistant(existing_llm=None)

    @Slot(str)
    def update_performance_mode(self, mode):
        """Legacy slot for backward compatibility."""
        n_layers = MODEL_TOTAL_LAYERS if mode.lower() == "high" else 24
        n_ctx = 40960 if mode.lower() == "high" else 8192
        self.update_model_settings(n_layers, n_ctx)

    @Slot(result=str)
    def get_vram_info(self):
        """Return JSON string with VRAM info + model estimation constants."""
        info = get_initial_vram_info()

        info.update({
            "total_layers": MODEL_TOTAL_LAYERS,
            "per_layer_mb": MODEL_PER_LAYER_MB,
            "kv_per_token_per_layer_mb": KV_PER_TOKEN_PER_LAYER_MB,
            "overhead_mb": VRAM_FIXED_OVERHEAD_MB,
            "ctx_min": CTX_MIN,
            "ctx_max": CTX_MAX,
        })
        return json.dumps(info)

    @Slot(result=str)
    def get_model_settings(self):
        """Return JSON string with current n_gpu_layers and n_ctx."""
        llm_cfg = config.get("llm", {})
        return json.dumps({
            "n_gpu_layers": llm_cfg.get("n_gpu_layers", 0),
            "n_ctx": llm_cfg.get("n_ctx", 8192),
        })

    @Slot(result=str)
    def get_performance_mode(self):
        """Legacy slot for backward compatibility."""
        return "high" if config.get("llm", {}).get("n_gpu_layers", 0) > 30 else "low"

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
