"""
VRAM probing and estimation utilities.
Uses llama_cpp to detect GPU support, then platform APIs (nvidia-smi, Windows
registry, Linux sysfs) to query actual VRAM capacity.
"""
import sys


def _probe_nvidia_smi() -> dict | None:
    """Query VRAM via nvidia-smi. Works on both Windows and Linux with NVIDIA drivers.
    Returns real-time total and free VRAM."""
    try:
        import subprocess
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(
            cmd, timeout=3, stderr=subprocess.DEVNULL
        ).decode().strip()
        first_line = out.splitlines()[0]
        parts = first_line.split(",")
        if len(parts) >= 2:
            total_mb = float(parts[0].strip())
            free_mb = float(parts[1].strip())
            return {
                "has_gpu": True,
                "gpu_type": "nvidia",
                "vram_monitored": True,
                "total_mb": total_mb,
                "free_mb": free_mb,
                "used_mb": round(total_mb - free_mb, 2),
            }
    except Exception:
        pass
    return None


def _probe_windows_registry() -> dict | None:
    """Query VRAM on Windows via nvidia-smi or DXGI registry key (64-bit, all GPU vendors).
    Reads HardwareInformation.qwMemorySize which is what Task Manager uses."""
    if sys.platform != "win32":
        return None

    # Try nvidia-smi first for real-time VRAM monitoring
    nv_info = _probe_nvidia_smi()
    if nv_info:
        return nv_info

    try:
        import winreg
        base_path = r"SYSTEM\ControlSet001\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        max_vram = 0
        for i in range(10):  # check up to 10 display adapters
            try:
                subkey = f"{base_path}\\{i:04d}"
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, subkey) as key:
                    val, _ = winreg.QueryValueEx(key, "HardwareInformation.qwMemorySize")
                    if isinstance(val, int) and val > max_vram:
                        max_vram = val
            except (FileNotFoundError, OSError):
                continue
        if max_vram > 0:
            total_mb = round(max_vram / (1024 * 1024), 2)
            # Registry only gives total size, not real-time free; estimate 90% free
            free_mb = round(total_mb * 0.90, 2)
            return {
                "has_gpu": True,
                "has_cuda": False,
                "gpu_type": "generic",
                "vram_monitored": False,
                "total_mb": total_mb,
                "free_mb": free_mb,
                "used_mb": round(total_mb - free_mb, 2),
            }
    except Exception:
        pass
    return None


_probe_win32_gpu_vram = _probe_windows_registry


def _probe_linux_sysfs() -> dict | None:
    """Query VRAM on Linux via nvidia-smi or DRM sysfs nodes (AMD / Intel).
    Provides real-time total and used VRAM."""
    if not sys.platform.startswith("linux"):
        return None

    # Try nvidia-smi first for real-time VRAM monitoring
    nv_info = _probe_nvidia_smi()
    if nv_info:
        return nv_info

    try:
        import glob
        import os

        vram_total_files = glob.glob("/sys/class/drm/card*/device/mem_info_vram_total")
        if not vram_total_files:
            return None

        tot_file = vram_total_files[0]
        used_file = tot_file.replace("mem_info_vram_total", "mem_info_vram_used")
        with open(tot_file, "r") as f:
            tot_bytes = int(f.read().strip())
        used_bytes = 0
        if os.path.exists(used_file):
            with open(used_file, "r") as f:
                used_bytes = int(f.read().strip())

        if tot_bytes > 0:
            total_mb = round(tot_bytes / (1024 * 1024), 2)
            free_mb = round((tot_bytes - used_bytes) / (1024 * 1024), 2)
            return {
                "has_gpu": True,
                "has_cuda": False,
                "gpu_type": "amd/intel",
                "vram_monitored": True,
                "total_mb": total_mb,
                "free_mb": free_mb,
                "used_mb": round(used_bytes / (1024 * 1024), 2),
            }
    except Exception:
        pass
    return None


_probe_linux_gpu_vram = _probe_linux_sysfs


def _get_platform_vram() -> dict | None:
    """Try all platform-specific methods to get GPU VRAM."""
    if sys.platform == "win32":
        return _probe_win32_gpu_vram()
    elif sys.platform.startswith("linux"):
        return _probe_linux_gpu_vram()
    return None


_INITIAL_VRAM_INFO = None


def get_initial_vram_info() -> dict:
    """
    Returns the VRAM info cached at application startup.
    This prevents 'free_mb' from shrinking as our own models are loaded,
    giving a consistent value for UI limits and proactive config checks.
    """
    global _INITIAL_VRAM_INFO
    if _INITIAL_VRAM_INFO is None:
        _INITIAL_VRAM_INFO = get_vram_info()
    return _INITIAL_VRAM_INFO


def get_vram_info() -> dict:
    """
    Returns GPU memory info.

    Detection strategy:
      1. torch.cuda (if available / mocked in tests)
      2. llama_cpp.llama_supports_gpu_offload() — does runtime support GPU?
      3. nvidia-smi / Windows registry / Linux sysfs

    Returns dict with keys: has_gpu, has_cuda, gpu_type, vram_monitored, total_mb, free_mb, used_mb.
    """
    no_gpu = {
        "has_gpu": False,
        "has_cuda": False,
        "gpu_type": "none",
        "vram_monitored": False,
        "total_mb": 0.0,
        "free_mb": 0.0,
        "used_mb": 0.0,
    }

    try:
        # Check torch.cuda if available (supports unittest mocking of torch)
        try:
            import torch
            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                total_mb = round(total_bytes / (1024 * 1024), 2)
                free_mb = round(free_bytes / (1024 * 1024), 2)
                used_mb = round((total_bytes - free_bytes) / (1024 * 1024), 2)
                return {
                    "has_gpu": True,
                    "has_cuda": True,
                    "gpu_type": "cuda",
                    "vram_monitored": True,
                    "total_mb": total_mb,
                    "free_mb": free_mb,
                    "used_mb": used_mb,
                }
        except (ImportError, Exception):
            pass

        # Step 1: ask llama.cpp if GPU offload is even compiled in
        supports_gpu = False
        try:
            import llama_cpp
            supports_gpu = llama_cpp.llama_supports_gpu_offload()
        except ImportError:
            pass

        if not supports_gpu:
            return no_gpu

        # Step 2: query actual VRAM via platform APIs
        vram = _get_platform_vram()
        if vram:
            if "has_cuda" not in vram:
                vram["has_cuda"] = (vram.get("gpu_type") in ("cuda", "nvidia"))
            return vram

        return no_gpu
    except Exception as e:
        print(f"[VRAM] Warning: Failed to query GPU memory: {e}")
        return no_gpu


def estimate_vram_usage(
    n_gpu_layers: int,
    n_ctx: int,
    per_layer_mb: float = 80.0,
    kv_per_token_per_layer_mb: float = 0.00012,
    overhead_mb: float = 300.0,
) -> dict:
    """
    Estimate VRAM required for a given layer count and context size.
    Returns breakdown dict: weights_mb, kv_mb, overhead_mb, total_mb.
    """
    if n_gpu_layers <= 0:
        return {
            "weights_mb": 0.0,
            "kv_mb": 0.0,
            "overhead_mb": 0.0,
            "total_mb": 0.0,
        }

    weights_mb = n_gpu_layers * per_layer_mb
    kv_mb = n_gpu_layers * n_ctx * kv_per_token_per_layer_mb
    total_mb = weights_mb + kv_mb + overhead_mb

    return {
        "weights_mb": round(weights_mb, 2),
        "kv_mb": round(kv_mb, 2),
        "overhead_mb": round(overhead_mb, 2),
        "total_mb": round(total_mb, 2),
    }


def estimate_max_gpu_layers(
    free_vram_mb: float,
    n_ctx: int,
    total_layers: int = 65,
    per_layer_mb: float = 80.0,
    kv_per_token_per_layer_mb: float = 0.00012,
    overhead_mb: float = 300.0,
) -> int:
    """
    Calculate the maximum number of GPU layers that safely fit in free VRAM
    for a given context length.
    """
    usable = free_vram_mb - overhead_mb
    if usable <= 0:
        return 0

    cost_per_layer = per_layer_mb + (n_ctx * kv_per_token_per_layer_mb)
    if cost_per_layer <= 0:
        return total_layers

    max_layers = int(usable / cost_per_layer)
    return max(0, min(max_layers, total_layers))


def estimate_max_context(
    free_vram_mb: float,
    n_gpu_layers: int,
    ctx_min: int = 2048,
    ctx_max: int = 40960,
    per_layer_mb: float = 80.0,
    kv_per_token_per_layer_mb: float = 0.00012,
    overhead_mb: float = 300.0,
) -> int:
    """
    Calculate the maximum context length that fits in free VRAM
    for a given GPU layer count.
    """
    if n_gpu_layers <= 0:
        return ctx_max

    weights_and_overhead = (n_gpu_layers * per_layer_mb) + overhead_mb
    remaining = free_vram_mb - weights_and_overhead
    if remaining <= 0:
        return ctx_min

    kv_cost_per_token = n_gpu_layers * kv_per_token_per_layer_mb
    if kv_cost_per_token <= 0:
        return ctx_max

    max_ctx = int(remaining / kv_cost_per_token)
    return max(ctx_min, min(max_ctx, ctx_max))
