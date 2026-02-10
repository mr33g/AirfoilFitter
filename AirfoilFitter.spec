# -*- mode: python ; coding: utf-8 -*-
import os
import sys
block_cipher = None

is_win = sys.platform == 'win32'
is_darwin = sys.platform == 'darwin'

icon_path = None
if is_win and os.path.exists(os.path.join('img', 'favicon.ico')):
    icon_path = os.path.join('img', 'favicon.ico')
elif is_darwin and os.path.exists(os.path.join('img', 'favicon.icns')):
    icon_path = os.path.join('img', 'favicon.icns')

analysis = Analysis(
    ['run_gui.py'],
    pathex=['.'],
    binaries=[],
    datas=[('img', 'img')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(analysis.pure, analysis.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name='AirfoilFitter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=icon_path,
)

if is_darwin:
    app = BUNDLE(
        exe,
        name='AirfoilFitter.app',
        icon=icon_path,
        bundle_identifier='com.michaelreeg.airfoilfitter',
    )
    coll = COLLECT(
        app,
        analysis.binaries,
        analysis.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='AirfoilFitter',
    )
else:
    coll = COLLECT(
        exe,
        analysis.binaries,
        analysis.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='AirfoilFitter',
    )
