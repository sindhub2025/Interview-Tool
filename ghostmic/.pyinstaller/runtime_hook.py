"""Runtime setup for PyInstaller-bundled GhostMic builds."""

import os


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


_base_dir = _ensure_dir(os.path.join(os.path.expanduser("~"), ".ghostmic"))

# Keep model caches in a predictable, writable directory for end users.
os.environ.setdefault("TORCH_HOME", _ensure_dir(os.path.join(_base_dir, "torch")))
os.environ.setdefault("HF_HOME", _ensure_dir(os.path.join(_base_dir, "hf")))