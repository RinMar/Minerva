"""
Tests for configuration management.
Verifies load_config, update_model_settings, save_model_settings, and save_last_user
correctly read, merge, and persist settings to config.toml.
"""
import os
import unittest
import tomllib
from unittest.mock import patch

import src.config as config_module
from src.config import (
    load_config, update_model_settings, save_model_settings,
    save_last_user, DEFAULT_CONFIG_TOML, MODEL_TOTAL_LAYERS
)


class TestLoadConfig(unittest.TestCase):
    """Tests for load_config and layer/context resolution."""

    def setUp(self):
        self.tmp_config = os.path.join(os.environ.get("TEMP", "/tmp"), "test_config.toml")
        with open(self.tmp_config, "w", encoding="utf-8") as f:
            f.write(DEFAULT_CONFIG_TOML)
        self.patcher = patch.object(config_module, "CONFIG_PATH", self.tmp_config)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        if os.path.exists(self.tmp_config):
            os.remove(self.tmp_config)

    def test_load_config_default_values(self):
        """Default llm configuration values."""
        cfg = load_config()
        self.assertEqual(cfg["llm"]["n_ctx"], 8192)
        self.assertEqual(cfg["llm"]["n_gpu_layers"], 0)

    def test_load_config_merges_model_ids(self):
        """Model IDs from the config file should be merged in."""
        cfg = load_config()
        self.assertIn("embedding_model_id", cfg["models"])
        self.assertIn("reranker_model_id", cfg["models"])
        self.assertIn("triplet_model_id", cfg["models"])


class TestUpdateModelSettings(unittest.TestCase):
    """Tests for update_model_settings (runtime switching)."""

    def setUp(self):
        self.tmp_config = os.path.join(os.environ.get("TEMP", "/tmp"), "test_config2.toml")
        with open(self.tmp_config, "w", encoding="utf-8") as f:
            f.write(DEFAULT_CONFIG_TOML)
        self.patcher = patch.object(config_module, "CONFIG_PATH", self.tmp_config)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        if os.path.exists(self.tmp_config):
            os.remove(self.tmp_config)

    def test_update_model_settings_updates_layers_and_ctx(self):
        """update_model_settings should update the global config dict."""
        result = update_model_settings(42, 16384)
        self.assertEqual(result["llm"]["n_gpu_layers"], 42)
        self.assertEqual(result["llm"]["n_ctx"], 16384)

    def test_update_model_settings_mutates_global(self):
        """The global config dict should be updated in-place."""
        update_model_settings(30, 32768)
        self.assertEqual(config_module.config["llm"]["n_gpu_layers"], 30)
        self.assertEqual(config_module.config["llm"]["n_ctx"], 32768)


class TestSaveModelSettings(unittest.TestCase):
    """Tests for save_model_settings (disk persistence)."""

    def setUp(self):
        self.tmp_config = os.path.join(os.environ.get("TEMP", "/tmp"), "test_config3.toml")
        with open(self.tmp_config, "w", encoding="utf-8") as f:
            f.write(DEFAULT_CONFIG_TOML)
        self.patcher = patch.object(config_module, "CONFIG_PATH", self.tmp_config)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        if os.path.exists(self.tmp_config):
            os.remove(self.tmp_config)

    def test_save_model_settings_persists(self):
        """save_model_settings should write settings to the config file."""
        save_model_settings(50, 20480)

        with open(self.tmp_config, "rb") as f:
            raw = tomllib.load(f)
        self.assertEqual(raw["llm"]["n_gpu_layers"], 50)
        self.assertEqual(raw["llm"]["n_ctx"], 20480)

    def test_save_model_settings_preserves_other_sections(self):
        """Saving model settings should not destroy other config sections."""
        save_model_settings(MODEL_TOTAL_LAYERS, 40960)

        with open(self.tmp_config, "rb") as f:
            raw = tomllib.load(f)
        self.assertIn("llm", raw)
        self.assertIn("user", raw)
        self.assertEqual(raw["user"]["last_user_name"], "user")


class TestSaveLastUser(unittest.TestCase):
    """Tests for save_last_user (disk persistence)."""

    def setUp(self):
        self.tmp_config = os.path.join(os.environ.get("TEMP", "/tmp"), "test_config4.toml")
        with open(self.tmp_config, "w", encoding="utf-8") as f:
            f.write(DEFAULT_CONFIG_TOML)
        self.patcher = patch.object(config_module, "CONFIG_PATH", self.tmp_config)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        if os.path.exists(self.tmp_config):
            os.remove(self.tmp_config)

    def test_save_last_user_persists(self):
        """save_last_user should write user_id and user_name to config file."""
        save_last_user(42, "alice")

        with open(self.tmp_config, "rb") as f:
            raw = tomllib.load(f)
        self.assertEqual(raw["user"]["last_user_id"], 42)
        self.assertEqual(raw["user"]["last_user_name"], "alice")

    def test_save_last_user_preserves_other_sections(self):
        """Saving a user should not destroy other sections."""
        save_last_user(7, "bob")

        with open(self.tmp_config, "rb") as f:
            raw = tomllib.load(f)
        self.assertIn("llm", raw)
        self.assertIn("user", raw)


if __name__ == "__main__":
    unittest.main()
