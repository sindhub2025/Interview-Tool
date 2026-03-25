# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules


hiddenimports = [
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "torch",
    "torchaudio",
    "faster_whisper",
    "ctranslate2",
    "onnxruntime",
    "pyaudiowpatch",
    "pynput",
    "sounddevice",
    "groq",
    "openai",
    "requests",
]

# faster_whisper relies on dynamic imports; include submodules explicitly.
hiddenimports += collect_submodules("faster_whisper")
hiddenimports += collect_submodules("ctranslate2")

binaries = []
binaries += collect_dynamic_libs("torch")
binaries += collect_dynamic_libs("torchaudio")
binaries += collect_dynamic_libs("onnxruntime")
binaries += collect_dynamic_libs("ctranslate2")
binaries += collect_dynamic_libs("av")

datas = [
    ("ghostmic/assets", "ghostmic/assets"),
    ("ghostmic/config.json", "ghostmic"),
]


a = Analysis(
    ["ghostmic/main.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["ghostmic/.pyinstaller/runtime_hook.py"],
    excludes=[
        "test",
        "tests",
        "unittest",
        "doctest",
        "setuptools",
        "pip",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="InterviewTool",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="ghostmic/assets/icon.ico",
)