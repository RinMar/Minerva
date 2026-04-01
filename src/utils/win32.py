"""
Win32 / DWM utility functions to manage low-level window attributes on Windows 11.
Used to remove forced 1px borders, rounded corners, and native shadows on frameless windows.
"""
import sys
import ctypes
from ctypes import wintypes

# DWM Attribute Constants
DWMWA_NCRENDERING_POLICY = 2
DWMWA_CAPTION_BUTTON_BOUNDS = 5
DWMWA_EXTEND_FRAME_INTO_CLIENT_AREA = 33  # Requires pvAttribute to be -1 Margin
DWMWA_WINDOW_CORNER_PREFERENCE = 33       # Use this for Windows 11 rounded corners
DWMWA_BORDER_COLOR = 34
DWMWA_COLOR_NONE = 0xFFFFFFFE

# Corner Preference Constants
DWMWCP_DEFAULT = 0
DWMWCP_DONOTROUND = 1
DWMWCP_ROUND = 2
DWMWCP_ROUNDSMALL = 3

# NCRendering Constants
DWMNCRP_DISABLED = 1
DWMNCRP_ENABLED = 2


def suppress_native_window_decorations(hwnd_int):
    """
    Surgically remove Windows 11 1px borders and rounded corners for a given window handle.
    """
    if sys.platform != "win32":
        return

    try:
        dwmapi = ctypes.windll.dwmapi
        hwnd = wintypes.HWND(hwnd_int)

        # 1. Disable 1px Accent Color Border
        border_color = wintypes.DWORD(DWMWA_COLOR_NONE)
        dwmapi.DwmSetWindowAttribute(
            hwnd,
            DWMWA_BORDER_COLOR,
            ctypes.byref(border_color),
            ctypes.sizeof(border_color)
        )

        # 2. Disable Rounded Corners (Windows 11)
        # Re-using DWMWA_WINDOW_CORNER_PREFERENCE which is 33
        corner_pref = wintypes.DWORD(DWMWCP_DONOTROUND)
        dwmapi.DwmSetWindowAttribute(
            hwnd,
            33,  # DWMWA_WINDOW_CORNER_PREFERENCE
            ctypes.byref(corner_pref),
            ctypes.sizeof(corner_pref)
        )

        # 3. Disable Non-Client Rendering entirely for this window
        nc_policy = wintypes.DWORD(DWMNCRP_DISABLED)
        dwmapi.DwmSetWindowAttribute(
            hwnd,
            DWMWA_NCRENDERING_POLICY,
            ctypes.byref(nc_policy),
            ctypes.sizeof(nc_policy)
        )

    except Exception as e:
        print(f"[Win32 Utility] Warning: Failed to set DWM attributes: {e}")
