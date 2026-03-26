"""Shared test fixtures for GhostMic."""

import os
import sys

# Ensure project root is on sys.path for test discovery
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
