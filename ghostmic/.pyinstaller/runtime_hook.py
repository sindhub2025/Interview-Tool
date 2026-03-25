"""Runtime setup for PyInstaller-bundled GhostMic builds."""

import os
import sys


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


_base_dir = _ensure_dir(os.path.join(os.path.expanduser("~"), ".ghostmic"))

# Keep model caches in a predictable, writable directory for end users.
os.environ.setdefault("TORCH_HOME", _ensure_dir(os.path.join(_base_dir, "torch")))
os.environ.setdefault("HF_HOME", _ensure_dir(os.path.join(_base_dir, "hf")))


def _add_windows_dll_paths() -> None:
    """Register bundled DLL directories for Windows frozen builds."""
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return

    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return

    candidate_dirs = [
        meipass,
        os.path.join(meipass, "torch", "lib"),
        os.path.join(meipass, "ctranslate2"),
        os.path.join(meipass, "onnxruntime"),
        os.path.join(meipass, "av"),
    ]

    for path in candidate_dirs:
        if os.path.isdir(path):
            try:
                os.add_dll_directory(path)
            except OSError:
                # Best-effort only; failures are logged by callers that load DLLs.
                pass


_add_windows_dll_paths()