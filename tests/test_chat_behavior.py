"""
Tests for conversational behavior.
Used to verify that the chat interface and model mock functionality work as expected.
"""
import unittest
from unittest.mock import MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.models.rag_chat import RAGChat
from src.memory.db import Base
import src.memory.db as db_module
from src.memory.db import EntityNode, GraphEdge
from src.memory.graph_manager import add_triplets
from src.models.embeddings import get_embedding_model
from src.chat import Chat
from src.models.base_llm import CustomLLM
from sqlalchemy.pool import StaticPool

# 1. Patch database engine for tests
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


class TestChatBehavior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db_module._engine = test_engine
        db_module.SessionLocal = TestSessionLocal
        Base.metadata.create_all(test_engine)

    def setUp(self):
        # Clear tables
        with db_module.get_session() as session:
            session.query(EntityNode).delete()
            session.query(GraphEdge).delete()
            session.commit()

        self.mock_llm = MagicMock()
        # Mock cross-encoder mapping inside the chat so it doesn't load a huge model
        self.mock_cross_encoder = MagicMock()
        self.mock_cross_encoder.predict.side_effect = lambda pairs: [0.9] * len(pairs)

    def test_mocked_llm_retrieve_flow(self):
        """Test that RAGChat correctly parses a <tool> block from the LLM, executes it, and feeds it back."""
        # Add mock data to the DB so the retrieve tool actually finds something
        add_triplets(
            [{"head": "User", "type": "is", "tail": "test user"}],
            "profile",
            ["identity"],
            "I am a test user.",
            "test_user",
            get_embedding_model()
        )

        # Yield a tool block first, then yield the final string
        def mock_generate(messages, **kwargs):
            last_msg = messages[-1]["content"]
            if "Tool execution result:" in last_msg:
                # This means rag_chat fed the tool result back to us!
                yield "Based on the retrieved context, you are a test user."
            else:
                # Output a retrieval tool call
                yield "<think>\nThinking...\n</think>\n"
                yield "<tool>\n"
                yield '{"name": "retrieve", "arguments": {"query": "who am I?"}}\n'
                yield "</tool>"

        self.mock_llm.generate.side_effect = mock_generate

        chat = RAGChat(user_name="test_user", llm=self.mock_llm)
        chat._cross_encoder = self.mock_cross_encoder

        response_tokens = list(chat.generate("who am I?", stream=True))
        final_answer = "".join(response_tokens)

        self.assertIn("Based on the retrieved context", final_answer)
        self.assertEqual(len(chat.history), 4)

        # Verify the mock data was populated and supplied to the LLM
        tool_result_msg = chat.history[2]["content"]
        self.assertIn("I am a test user.", tool_result_msg)

    def test_mocked_llm_manage_memory_flow(self):
        """Test that RAGChat successfully intercepts manage_memory arrays and triggers orchestrator."""
        def mock_generate(messages, **kwargs):
            last_msg = messages[-1]["content"]
            if "Tool execution result:" in last_msg:
                yield "Memory stored successfully."
                return

            yield "<think>\nStoring memory...\n</think>\n"
            yield "<tool>\n"
            payload = (
                '{"name": "manage_memory", "arguments": '
                '{"store": [{"topic": "hobbies", "fact": "user likes tennis", "tags": ["sport"], '
                '"triplets": [{"head": "user", "type": "likes", "tail": "tennis"}]}]}}\n'
            )
            yield payload
            yield "</tool>"

        self.mock_llm.generate.side_effect = mock_generate

        chat = RAGChat(user_name="test_user", llm=self.mock_llm)
        chat._cross_encoder = self.mock_cross_encoder

        # Spy on the orchestrator
        chat.orchestrator.trigger_store = MagicMock()

        list(chat.generate("I like tennis.", stream=True))

        # Verify the orchestrator was triggered securely
        chat.orchestrator.trigger_store.assert_called_once_with(
            "hobbies", fact="user likes tennis", triplets=[{"head": "user", "type": "likes", "tail": "tennis"}]
        )

    def test_chat_api_send_message(self):
        """Test the public Chat API promise from the README."""

        # Mock generator mimicking LLM responses
        def mock_generate(messages, **kwargs):
            yield "Hello "
            yield "Minerva!"

        self.mock_llm.generate.side_effect = mock_generate

        assistant = Chat(user_name="api_user", llm=self.mock_llm)
        assistant._cross_encoder = self.mock_cross_encoder

        response_tokens = list(assistant.send_message("Hello!", stream=True))
        final_answer = "".join(response_tokens)

        self.assertEqual(final_answer, "Hello Minerva!")

    def test_live_llm_minimal_e2e(self):
        """
        One step end-to-end test with real LLMs. Minimal processing to save compute.
        We initialize RAGChat normally, trigger a tiny stream, and break early.
        """
        # Initialize with real models (loads local LLMs)
        tiny_llm = CustomLLM(
            repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            filename="qwen2.5-0.5b-instruct-q8_0.gguf",
            n_ctx=2048,       # Increased to fit the initial prompt of ~1200 tokens
            use_mlock=False   # Avoid mlock limits on Linux CI runners
        )

        chat = RAGChat(user_name="e2e_user", llm=tiny_llm)

        generator = chat.generate("hi", stream=True)
        # Pull exactly one token to confirm the models loaded and context successfully initialized
        try:
            first_token = next(generator)
            self.assertIsInstance(first_token, str)
            self.assertTrue(len(first_token) >= 0)
        except StopIteration:
            self.fail("Real LLM failed to yield any tokens.")
