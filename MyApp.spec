# -*- mode: python ; coding: utf-8 -*-

import os
from dotenv import load_dotenv

# Load .env at build time
load_dotenv(".env")
api_key = os.getenv("API_KEY", "")

a = Analysis(
    ['ui.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('assets/', 'assets'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='App',
    debug=False,
    bootloader_ignore_signals=False,
    icon='icon.ico',
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=True,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# Bake the API key into the exe object
exe.api_key = api_key
