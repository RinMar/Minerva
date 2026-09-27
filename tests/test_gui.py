import unittest
from unittest.mock import MagicMock, patch
import json

from PySide6.QtCore import QCoreApplication

import src.chat
from src.gui.bridge import Bridge  # noqa: E402

# Ensure an application instance exists for QObject signals to work
app = QCoreApplication.instance()
if app is None:
    app = QCoreApplication([])


class MockPage:
    def __init__(self):
        self.scripts = []

    def runJavaScript(self, script):
        self.scripts.append(script)


class TestGUIBridge(unittest.TestCase):
    def setUp(self):
        self.mock_page = MockPage()

        # We patch Chat to avoid loading heavy LLM models during GUI tests
        with patch('src.chat.Chat') as mock_chat_class:
            self.mock_chat_instance = MagicMock()
            mock_chat_class.return_value = self.mock_chat_instance

            self.bridge = Bridge(page=self.mock_page, user_id=1, user_name="test_gui_user")
            self.bridge.assistant = self.mock_chat_instance

    def test_bridge_initialization(self):
        self.assertEqual(self.bridge.user_name, "test_gui_user")
        self.assertEqual(self.bridge.page, self.mock_page)

    def test_push_assistant_message(self):
        test_msg = "Hello, world!"
        self.bridge._push_assistant_message(test_msg)

        expected_js = f"appendMessage('assistant', {json.dumps(test_msg)})"
        self.assertIn(expected_js, self.mock_page.scripts)

    def test_push_graph(self):
        test_data = {"nodes": [{"id": "1", "label": "Node"}], "edges": []}
        self.bridge.push_graph(test_data)

        expected_js = f"updateGraph({json.dumps(test_data)})"
        self.assertIn(expected_js, self.mock_page.scripts)

    @patch('src.gui.bridge.threading.Thread')
    def test_send_message_starts_thread(self, mock_thread_class):
        mock_thread_instance = MagicMock()
        mock_thread_class.return_value = mock_thread_instance

        self.bridge.send_message("Hello AI")

        mock_thread_class.assert_called_once()
        args, kwargs = mock_thread_class.call_args
        self.assertEqual(kwargs['target'], self.bridge._generate_response)
        self.assertEqual(kwargs['args'], ("Hello AI",))
        mock_thread_instance.start.assert_called_once()

    def test_get_vram_info_slot(self):
        vram_json = self.bridge.get_vram_info()
        data = json.loads(vram_json)
        self.assertIn("has_gpu", data)
        self.assertIn("total_layers", data)
        self.assertIsInstance(data["total_layers"], int)

    def test_get_model_settings_slot(self):
        with patch.dict('src.config.config', {"llm": {"n_gpu_layers": 42, "n_ctx": 16384}}, clear=False):
            settings_json = self.bridge.get_model_settings()
            data = json.loads(settings_json)
            self.assertEqual(data["n_gpu_layers"], 42)
            self.assertEqual(data["n_ctx"], 16384)

    @patch('src.models.embeddings.reset_models')
    @patch('src.gui.bridge.update_model_settings')
    def test_update_model_settings_calls_pipeline(self, mock_update, mock_reset):
        self.bridge.initialize_assistant = MagicMock()

        self.bridge.update_model_settings(30, 8192)

        mock_update.assert_called_once_with(30, 8192, model_key=None)
        mock_reset.assert_called_once()
        self.bridge.initialize_assistant.assert_called_once_with(existing_llm=None)

    def test_send_message_blocked_without_assistant(self):
        self.bridge.assistant = None
        self.mock_page.scripts.clear()

        self.bridge.send_message("Should be ignored")

        thread_related = [s for s in self.mock_page.scripts if "generate" in s.lower()]
        self.assertEqual(len(thread_related), 0)

    def test_bg_init_assistant_hides_loader_on_success(self):
        with patch('src.chat.Chat') as mock_chat:
            mock_chat.return_value = MagicMock()

            statuses = []
            self.bridge.model_loading_status.connect(
                lambda loading, msg: statuses.append((loading, msg))
            )

            self.bridge._bg_init_assistant()

            self.assertTrue(any(s[0] is True for s in statuses))
            self.assertEqual(statuses[-1], (False, ""))

    def test_bg_init_assistant_retry_on_oom(self):
        """When initial Chat fails, _bg_init_assistant should attempt fallback load with safe layers."""
        with patch('src.chat.Chat', side_effect=[RuntimeError("CUDA out of memory"), MagicMock()]):
            with patch('src.gui.bridge.get_initial_vram_info', return_value={"free_mb": 2000.0}):
                with patch('src.models.base_llm.CustomLLM') as mock_custom_llm:
                    mock_custom_llm.return_value = MagicMock()

                    warnings = []
                    self.bridge.model_warning.connect(lambda msg: warnings.append(msg))

                    self.bridge._bg_init_assistant()

                    # Fallback retry should have been attempted
                    mock_custom_llm.assert_called_once()
                    self.assertTrue(len(warnings) > 0)

    @patch('src.gui.bridge.save_last_user')
    def test_delete_user_switching_and_fallback(self, mock_save_user):
        """Test deleting active/inactive users and fallback to default user when last user is deleted."""
        from src.memory.db import get_session, User, EntityNode, GraphEdge
        
        # Create 3 users in test DB
        with get_session() as session:
            session.query(User).delete()
            u1 = User(name="User1")
            u2 = User(name="User2")
            u3 = User(name="User3")
            session.add_all([u1, u2, u3])
            session.commit()
            u1_id, u2_id, u3_id = u1.id, u2.id, u3.id

        self.bridge.initialize_assistant = MagicMock()

        # Case 1: Delete non-active user (User3 when active is User2)
        self.bridge.user_id = u2_id
        self.bridge.user_name = "User2"
        self.bridge.delete_user("User3")

        with get_session() as session:
            remaining = [u.name for u in session.query(User).order_by(User.id).all()]
            self.assertEqual(remaining, ["User1", "User2"])
        self.assertEqual(self.bridge.user_name, "User2")

        # Case 2: Delete active user (User2 when active is User2) -> switches to User1 (above it)
        self.bridge.delete_user("User2")

        with get_session() as session:
            remaining = [u.name for u in session.query(User).order_by(User.id).all()]
            self.assertEqual(remaining, ["User1"])
        self.assertEqual(self.bridge.user_name, "User1")

        # Case 3: Delete last remaining user (User1) -> creates fresh default User profile
        self.bridge.delete_user("User1")

        with get_session() as session:
            remaining = [u.name for u in session.query(User).order_by(User.id).all()]
            self.assertEqual(remaining, ["User"])
        self.assertEqual(self.bridge.user_name, "User")


if __name__ == "__main__":
    unittest.main()
