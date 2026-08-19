# -*- mode: python ; coding: utf-8 -*-

import hashlib
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)


def _source_signature() -> str:
    digest = hashlib.sha256()
    root = Path.cwd()
    paths = sorted((root / "ghostmic").rglob("*.py")) + [root / "InterviewTool.spec"]
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


signature_path = Path("build") / "InterviewTool.source_signature"
signature_path.parent.mkdir(parents=True, exist_ok=True)
signature_path.write_text(_source_signature(), encoding="ascii")


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
    (str(signature_path), "ghostmic"),
]
datas += collect_data_files(
    "faster_whisper",
    includes=["assets/*.onnx"],
)

try:
    import torch

    cached_vad_model = (
        Path(torch.hub.get_dir())
        / "snakers4_silero-vad_master"
        / "src"
        / "silero_vad"
        / "data"
        / "silero_vad.jit"
    )
    if cached_vad_model.is_file():
        datas.append((str(cached_vad_model), "ghostmic/assets"))
except Exception:
    pass


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
