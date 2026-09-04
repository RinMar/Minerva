"""
Unit tests for src.utils.vram module.
"""
from unittest.mock import patch
from src.utils.vram import (
    get_vram_info,
    estimate_vram_usage,
    estimate_max_gpu_layers,
    estimate_max_context,
)


def test_estimate_vram_usage_zero_layers():
    usage = estimate_vram_usage(0, 8192)
    assert usage["total_mb"] == 0.0
    assert usage["weights_mb"] == 0.0


def test_estimate_vram_usage_positive_layers():
    usage = estimate_vram_usage(n_gpu_layers=30, n_ctx=8192)
    # weights: 30 * 80 = 2400
    # kv: 30 * 8192 * 0.00012 = 29.4912
    # overhead: 300
    assert usage["weights_mb"] == 2400.0
    assert usage["overhead_mb"] == 300.0
    assert usage["total_mb"] == 2729.49


def test_estimate_max_gpu_layers():
    # 5200 MB free, 8192 ctx -> usable = 4900 MB
    # cost per layer = 80 + 8192 * 0.00012 = 80.98304
    # max = 4900 / 80.98304 = ~60 layers
    max_l = estimate_max_gpu_layers(free_vram_mb=5200, n_ctx=8192, total_layers=65)
    assert max_l == 60


def test_estimate_max_gpu_layers_insufficient_vram():
    # Only 200 MB free -> less than overhead 300 -> 0 layers
    max_l = estimate_max_gpu_layers(free_vram_mb=200, n_ctx=8192, total_layers=65)
    assert max_l == 0


def test_estimate_max_context():
    # 5200 MB free, 60 layers
    # weights + overhead = 60 * 80 + 300 = 5100
    # remaining = 100 MB
    # kv_cost_per_token = 60 * 0.00012 = 0.0072
    # max_ctx = 100 / 0.0072 = ~13888
    max_c = estimate_max_context(free_vram_mb=5200, n_gpu_layers=60)
    assert 13000 <= max_c <= 14500


def test_get_vram_info_mocked():
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.mem_get_info", return_value=(4 * 1024**3, 8 * 1024**3)):
            info = get_vram_info()
            assert info["has_cuda"] is True
            assert info["total_mb"] == 8192.0
            assert info["free_mb"] == 4096.0
            assert info["used_mb"] == 4096.0


def test_get_vram_info_no_cuda():
    with patch("torch.cuda.is_available", return_value=False):
        with patch("src.utils.vram._probe_win32_gpu_vram", return_value=None):
            with patch("src.utils.vram._probe_linux_gpu_vram", return_value=None):
                info = get_vram_info()
                assert info["has_gpu"] is False
                assert info["has_cuda"] is False
                assert info["free_mb"] == 0.0
