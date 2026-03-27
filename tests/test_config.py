"""
Tests for configuration management.
Verifies load_config, update_config_mode, save_performance_mode, and save_last_user
correctly read, merge, and persist settings to config.toml.
"""
import os
import unittest
import tomllib
from unittest.mock import patch

import src.config as config_module
from src.config import (
    load_config, update_config_mode, save_performance_mode,
    save_last_user, DEFAULT_CONFIG_TOML
)


class TestLoadConfig(unittest.TestCase):
    """Tests for load_config and performance mode resolution."""

    def setUp(self):
        self.tmp_config = os.path.join(os.environ.get("TEMP", "/tmp"), "test_config.toml")
        # Write default config to temp file
        with open(self.tmp_config, "w", encoding="utf-8") as f:
            f.write(DEFAULT_CONFIG_TOML)
        # Patch CONFIG_PATH to use our temp file
        self.patcher = patch.object(config_module, "CONFIG_PATH", self.tmp_config)
        self.patcher.start()
        # Clear the env var so it doesn't interfere
        os.environ.pop("MINERVA_PERFORMANCE", None)

    def tearDown(self):
        self.patcher.stop()
        os.environ.pop("MINERVA_PERFORMANCE", None)
        if os.path.exists(self.tmp_config):
            os.remove(self.tmp_config)

    def test_load_config_default_mode_is_low(self):
        """Default performance mode should be 'low' when nothing is set."""
        cfg = load_config()
        self.assertEqual(cfg["performance_mode"], "low")

    def test_load_config_low_preset_values(self):
        """Low performance preset should apply n_ctx=8192 and device=cpu."""
        cfg = load_config()
        self.assertEqual(cfg["llm"]["n_ctx"], 8192)
        self.assertEqual(cfg["llm"]["device"], "cpu")
        self.assertEqual(cfg["llm"]["n_gpu_layers"], 24)

    def test_load_config_high_preset_via_env(self):
        """Setting MINERVA_PERFORMANCE=high should apply the high preset."""
        os.environ["MINERVA_PERFORMANCE"] = "high"
        cfg = load_config()
        self.assertEqual(cfg["performance_mode"], "high")
        self.assertEqual(cfg["llm"]["n_ctx"], 40960)
        self.assertEqual(cfg["llm"]["device"], "cuda")
        self.assertEqual(cfg["llm"]["n_gpu_layers"], -1)

    def test_load_config_env_overrides_file(self):
        """Environment variable should take priority over config file setting."""
        save_performance_mode("low")
        os.environ["MINERVA_PERFORMANCE"] = "high"
        cfg = load_config()
        self.assertEqual(cfg["performance_mode"], "high")

    def test_load_config_reads_file_setting(self):
        """Config file [settings].performance_mode should be used when env is not set."""
        save_performance_mode("high")
        cfg = load_config()
        self.assertEqual(cfg["performance_mode"], "high")

    def test_load_config_merges_model_ids(self):
        """Model IDs from the config file should be merged in."""
        cfg = load_config()
        self.assertIn("embedding_model_id", cfg["models"])
        self.assertIn("reranker_model_id", cfg["models"])
        self.assertIn("triplet_model_id", cfg["models"])


class TestUpdateConfigMode(unittest.TestCase):
    """Tests for update_config_mode (runtime switching)."""

    def setUp(self):
        self.tmp_config = os.path.join(os.environ.get("TEMP", "/tmp"), "test_config2.toml")
        with open(self.tmp_config, "w", encoding="utf-8") as f:
            f.write(DEFAULT_CONFIG_TOML)
        self.patcher = patch.object(config_module, "CONFIG_PATH", self.tmp_config)
        self.patcher.start()
        os.environ.pop("MINERVA_PERFORMANCE", None)

    def tearDown(self):
        self.patcher.stop()
        os.environ.pop("MINERVA_PERFORMANCE", None)
        if os.path.exists(self.tmp_config):
            os.remove(self.tmp_config)

    def test_update_config_mode_switches_to_high(self):
        """update_config_mode('high') should update the global config dict."""
        result = update_config_mode("high")
        self.assertEqual(result["performance_mode"], "high")
        self.assertEqual(result["llm"]["device"], "cuda")

    def test_update_config_mode_sets_env_var(self):
        """update_config_mode should set the MINERVA_PERFORMANCE env var."""
        update_config_mode("high")
        self.assertEqual(os.environ["MINERVA_PERFORMANCE"], "high")

    def test_update_config_mode_mutates_global(self):
        """The global config dict should be updated in-place."""
        update_config_mode("high")
        self.assertEqual(config_module.config["performance_mode"], "high")
        self.assertEqual(config_module.config["llm"]["device"], "cuda")


class TestSavePerformanceMode(unittest.TestCase):
    """Tests for save_performance_mode (disk persistence)."""

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

    def test_save_performance_mode_persists(self):
        """save_performance_mode should write the mode to the config file."""
        save_performance_mode("high")

        with open(self.tmp_config, "rb") as f:
            raw = tomllib.load(f)
        self.assertEqual(raw["settings"]["performance_mode"], "high")

    def test_save_performance_mode_preserves_other_sections(self):
        """Saving performance mode should not destroy other config sections."""
        save_performance_mode("high")

        with open(self.tmp_config, "rb") as f:
            raw = tomllib.load(f)
        # [llm] and [user] sections should still exist
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
        self.assertIn("settings", raw)


if __name__ == "__main__":
    unittest.main()
