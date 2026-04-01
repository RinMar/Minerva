"""
Tests for the embeddings module.
Verifies reset_models, get_device, and the lazy-loading behavior
of get_embedding_model and get_reranker_model.
"""
import unittest
from unittest.mock import patch, MagicMock

import src.models.embeddings as emb_module


class TestGetDevice(unittest.TestCase):
    """Tests for get_device()."""

    @patch("torch.cuda.is_available")
    def test_get_device_falls_back_to_cpu_if_no_cuda(self, mock_cuda):
        """get_device should fall back to 'cpu' if 'cuda' is requested but not available."""
        mock_cuda.return_value = False
        with patch.dict(emb_module.config, {"llm": {"device": "cuda"}}, clear=False):
            self.assertEqual(emb_module.get_device(), "cpu")

    def test_get_device_defaults_to_cpu(self):
        """get_device should default to 'cpu' if 'device' key is missing."""
        with patch.dict(emb_module.config, {"llm": {}}, clear=False):
            self.assertEqual(emb_module.get_device(), "cpu")


class TestResetModels(unittest.TestCase):
    """Tests for reset_models()."""

    def test_reset_models_clears_cache(self):
        """reset_models should set both module-level caches to None."""
        emb_module._embedding_model = "fake_model"
        emb_module._reranker_model = "fake_reranker"

        emb_module.reset_models()

        self.assertIsNone(emb_module._embedding_model)
        self.assertIsNone(emb_module._reranker_model)


class TestModelLifecycle(unittest.TestCase):
    """Tests for get_embedding_model and get_reranker_model singleton behavior."""

    def setUp(self):
        # Always start with a clean state
        emb_module._embedding_model = None
        emb_module._reranker_model = None

    def tearDown(self):
        emb_module._embedding_model = None
        emb_module._reranker_model = None

    @patch("src.models.embeddings.SentenceTransformer")
    def test_get_embedding_model_loads_once(self, mock_st):
        """get_embedding_model should only create the model once (lazy singleton)."""
        mock_st.return_value = MagicMock()

        model1 = emb_module.get_embedding_model()
        model2 = emb_module.get_embedding_model()

        mock_st.assert_called_once()
        self.assertIs(model1, model2)

    @patch("src.models.embeddings.SentenceTransformer")
    def test_get_embedding_model_reloads_after_reset(self, mock_st):
        """After reset_models(), get_embedding_model should create a new instance."""
        mock_st.return_value = MagicMock()

        model1 = emb_module.get_embedding_model()
        emb_module.reset_models()

        mock_st.return_value = MagicMock()
        model2 = emb_module.get_embedding_model()

        self.assertEqual(mock_st.call_count, 2)
        self.assertIsNot(model1, model2)

    @patch("src.models.embeddings.CrossEncoder")
    def test_get_reranker_model_loads_once(self, mock_ce):
        """get_reranker_model should only create the model once (lazy singleton)."""
        mock_ce.return_value = MagicMock()

        model1 = emb_module.get_reranker_model()
        model2 = emb_module.get_reranker_model()

        mock_ce.assert_called_once()
        self.assertIs(model1, model2)

    @patch("src.models.embeddings.CrossEncoder")
    def test_get_reranker_model_reloads_after_reset(self, mock_ce):
        """After reset_models(), get_reranker_model should create a new instance."""
        mock_ce.return_value = MagicMock()

        model1 = emb_module.get_reranker_model()
        emb_module.reset_models()

        mock_ce.return_value = MagicMock()
        model2 = emb_module.get_reranker_model()

        self.assertEqual(mock_ce.call_count, 2)
        self.assertIsNot(model1, model2)


if __name__ == "__main__":
    unittest.main()
