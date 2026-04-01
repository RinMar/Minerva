"""
Helper script to convert SVG logo to ICO format for Windows executables.
Requires PySide6 to be installed in the environment.
"""
import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPixmap


def convert_svg_to_ico(svg_path, ico_path, size=256):
    if not os.path.exists(svg_path):
        print(f"Error: Source file '{svg_path}' not found.")
        return False

    # QApplication is required for QPixmap to handle SVG via the SVG plugin
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    pixmap = QPixmap(svg_path)
    if pixmap.isNull():
        print(f"Error: Failed to load SVG from '{svg_path}'.")
        return False

    # Scale to typical high-res icon size
    pixmap = pixmap.scaled(size, size)

    if pixmap.save(ico_path, "ICO"):
        print(f"Successfully converted '{svg_path}' to '{ico_path}'")
        return True
    else:
        print(f"Error: Failed to save ICO to '{ico_path}'.")
        return False


if __name__ == "__main__":
    # Default paths for Minerva
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_svg = os.path.join(base_dir, "resources", "logo.svg")
    default_ico = os.path.join(base_dir, "resources", "logo.ico")

    svg = sys.argv[1] if len(sys.argv) > 1 else default_svg
    ico = sys.argv[2] if len(sys.argv) > 2 else default_ico

    convert_svg_to_ico(svg, ico)
