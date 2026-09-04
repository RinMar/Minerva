"""
Minerva Utility Package.
Provides submodules for specific domains (win32, general, etc.).
"""
from src.utils.general import cosine_similarity, fact_id
from src.utils.win32 import suppress_native_window_decorations
from src.utils.vram import (
    get_vram_info,
    estimate_vram_usage,
    estimate_max_gpu_layers,
    estimate_max_context,
)

__all__ = [
    'cosine_similarity',
    'fact_id',
    'suppress_native_window_decorations',
    'get_vram_info',
    'estimate_vram_usage',
    'estimate_max_gpu_layers',
    'estimate_max_context',
]
