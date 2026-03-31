# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Collect any additional data files from heavy packages if needed
datas = [
    ('resources', 'resources'),
    ('config.toml', '.'),
    ('.venv/Lib/site-packages/llama_cpp/lib', 'llama_cpp/lib'),
    ('src/memory/alembic', 'src/memory/alembic'),
]

# Specifically collect PySide6 WebEngine data if PyInstaller hooks miss them
# datas += collect_data_files('PySide6')

a = Analysis(
    ['src/run.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'PySide6.QtWebEngineWidgets',
        'PySide6.QtWebChannel',
        'sqlalchemy.sql.default_comparator',
        'logging.config',
        'logging.handlers',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tests', 'docs', '.pytest_cache', '__pycache__'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Minerva',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon=['resources/gui/icon.ico'], # Uncomment if you have an icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Minerva',
)
