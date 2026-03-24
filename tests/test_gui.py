import unittest
from unittest.mock import MagicMock, patch
import json

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from src.gui.main import Bridge

# Ensure an application instance exists for QObject signals to work
app = QApplication.instance()
if app is None:
    app = QApplication([])


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

            self.bridge = Bridge(page=self.mock_page, user_name="test_gui_user")

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
        signal_emitted = []

        def on_response(text):
            signal_emitted.append(text)

        self.bridge.response_ready.connect(on_response)

        # Call the private method directly (simulating what the thread does)
        self.bridge._generate_response("Test prompt")

        # Wait for signals to process via event loop
        QCoreApplication.processEvents()

        # Verify the think block was removed and the text was emitted
        self.assertEqual(len(signal_emitted), 1)
        self.assertEqual(signal_emitted[0], "Final answer.")
