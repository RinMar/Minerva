"""
Custom Splash Screen widget for Minerva.
Provides a perfectly transparent, DPI-aware logo display using QSvgRenderer.
"""
import sys
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QScreen, QImage
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtCore import Qt, QRectF
from src.utils.win32 import suppress_native_window_decorations


class SplashWidget(QWidget):
    """
    A custom frameless splash screen that renders an SVG logo with perfect transparency.
    Uses a Premultiplied ARGB32 pipeline and safety margins to eliminate edge artifacts.
    """
    def __init__(self, logo_path, base_width=400):
        super().__init__()

        # 1. Load SVG and measure native aspect ratio
        self.renderer = QSvgRenderer(logo_path)
        native_size = self.renderer.defaultSize()
        aspect_ratio = native_size.height() / native_size.width()

        # 2. Calculate logo dimensions
        self.logo_width = base_width
        self.logo_height = int(base_width * aspect_ratio)

        # 3. Add a 10px safety margin to isolate the logo from the physical window edge
        self.margin = 10
        window_width = self.logo_width + (self.margin * 2)
        window_height = self.logo_height + (self.margin * 2)

        # 4. Setup Window Flags: Tailor for each platform
        if sys.platform == "win32":
            # Aggressive flags to bypass Windows 11 DWM decorations
            self.setWindowFlags(
                Qt.WindowStaysOnTopHint |
                Qt.FramelessWindowHint |
                Qt.NoDropShadowWindowHint |
                Qt.WindowTransparentForInput |
                Qt.X11BypassWindowManagerHint
            )
        else:
            # Standard Qt approach for Linux/macOS
            self.setWindowFlags(
                Qt.SplashScreen |
                Qt.WindowStaysOnTopHint |
                Qt.FramelessWindowHint
            )

        # 5. Enable Translucency
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setStyleSheet("background: transparent; border: none;")

        # 6. Initialize Image Cache
        self._cached_image = None

        # 7. Sizing and Positioning
        self.resize(window_width, window_height)
        self.center_on_screen()

        # 8. Win32 Overwolf-style Fix: Surgical DWM Attribute Suppression
        # We pass the internal winId as an integer to the Win32 utility
        suppress_native_window_decorations(int(self.winId()))

    def center_on_screen(self):
        """Center the splash widget on the primary screen."""
        screen = QScreen.availableGeometry(self.screen())
        cp = screen.center()
        self.move(cp.x() - self.width() // 2, cp.y() - self.height() // 2)

    def paintEvent(self, event):
        """Render the logo using a Premultiplied ARGB32 buffer to avoid fringing."""
        if self._cached_image is None:
            # Create a high-DPI aware premultiplied image buffer
            dpr = self.devicePixelRatio()
            img_size = (int(self.logo_width * dpr), int(self.logo_height * dpr))

            # ARGB32_Premultiplied is crucial for correct alpha math on transparent bases
            self._cached_image = QImage(img_size[0], img_size[1], QImage.Format_ARGB32_Premultiplied)
            self._cached_image.fill(Qt.transparent)
            self._cached_image.setDevicePixelRatio(dpr)

            img_painter = QPainter(self._cached_image)
            img_painter.setRenderHint(QPainter.Antialiasing)
            self.renderer.render(img_painter)
            img_painter.end()

        # Final draw to screen with the 10px margin
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        target_rect = QRectF(self.margin, self.margin, self.logo_width, self.logo_height)
        painter.drawImage(target_rect, self._cached_image)
        painter.end()

    def finish(self, main_window):
        """Close the splash screen and transfer focus to the main window."""
        self.close()
        if main_window:
            main_window.setFocus()
            main_window.activateWindow()
            main_window.raise_()
