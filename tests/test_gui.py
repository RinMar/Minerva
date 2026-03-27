import unittest
from unittest.mock import MagicMock, patch
import json

from PySide6.QtCore import QCoreApplication

from src.gui.main import Bridge

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
        with patch('src.gui.main.Chat') as mock_chat_class:
            self.mock_chat_instance = MagicMock()
            mock_chat_class.return_value = self.mock_chat_instance

            self.bridge = Bridge(page=self.mock_page, user_id=1, user_name="test_gui_user")
            # Set the mock assistant directly as initialization is now deferred to a background thread
            self.bridge.assistant = self.mock_chat_instance

    def test_bridge_initialization(self):
        self.assertEqual(self.bridge.user_name, "test_gui_user")
        self.assertEqual(self.bridge.page, self.mock_page)

    def test_push_assistant_message(self):
        test_msg = "Hello, world!"
        self.bridge._push_assistant_message(test_msg)

        # Check if the correct javascript was constructed
        expected_js = f"appendMessage('assistant', {json.dumps(test_msg)})"
        self.assertIn(expected_js, self.mock_page.scripts)

    def test_push_graph(self):
        test_data = {"nodes": [{"id": "1", "label": "Node"}], "edges": []}
        self.bridge.push_graph(test_data)

        expected_js = f"updateGraph({json.dumps(test_data)})"
        self.assertIn(expected_js, self.mock_page.scripts)

    @patch('src.gui.main.threading.Thread')
    def test_send_message_starts_thread(self, mock_thread_class):
        mock_thread_instance = MagicMock()
        mock_thread_class.return_value = mock_thread_instance

        self.bridge.send_message("Hello AI")

        # Verify thread was created and started
        mock_thread_class.assert_called_once()
        args, kwargs = mock_thread_class.call_args
        self.assertEqual(kwargs['target'], self.bridge._generate_response)
        self.assertEqual(kwargs['args'], ("Hello AI",))
        mock_thread_instance.start.assert_called_once()

    def test_generate_response(self):
        # Setup mock generation to yield some tokens including a think tag
        def mock_generate(*args, **kwargs):
            yield "<think>\nThinking process...\n</think>\n"
            yield "Final "
            yield "answer."

        self.mock_chat_instance.send_message.side_effect = mock_generate

        # Connect to signal to verify it's emitted properly
        tokens_emitted = []

        def on_token(text):
            tokens_emitted.append(text)

        self.bridge.stream_token.connect(on_token)

        # Call the private method directly (simulating what the thread does)
        self.bridge._generate_response("Test prompt")

        # Wait for signals to process via event loop
        QCoreApplication.processEvents()

        # Verify the think block was removed and the text was emitted
        self.assertEqual("".join(tokens_emitted), "Final answer.")

    def test_generate_response_complex_flow(self):
        # Setup mock generation testing actions and stray </think> tags
        def mock_generate_complex(*args, **kwargs):
            yield "<think>\nInitial thought\n</think>\n"
            yield "Wait, I need to check something.\n"
            yield "<action:retrieve>"
            yield "</action:retrieve>"
            yield "<think>\nSecond thought\n</think>\n"
            yield "I found it. "
            yield "</think>\n"  # Stray tag that used to cause leaks
            yield "Final "
            yield "answer."

        self.mock_chat_instance.send_message.side_effect = mock_generate_complex

        tokens_emitted = []
        states_emitted = []

        def on_token(text):
            tokens_emitted.append(text)

        def on_state(state):
            states_emitted.append(state)

        # Disconnect previous test connections if they persist across run
        try:
            self.bridge.stream_token.disconnect()
            self.bridge.stream_state.disconnect()
        except RuntimeError:
            pass

        self.bridge.stream_token.connect(on_token)
        self.bridge.stream_state.connect(on_state)

        self.bridge._generate_response("Complex Test prompt")
        QCoreApplication.processEvents()

        expected_text = "Wait, I need to check something.\nI found it. \nFinal answer."
        self.assertEqual("".join(tokens_emitted), expected_text)

        self.assertIn("action_start_retrieve", states_emitted)
        self.assertIn("action_end", states_emitted)
