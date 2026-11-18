from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class CaptureRequest:
    target: str = "full_virtual_desktop"   # full_virtual_desktop | primary | monitor_N | active_window
    session_id: Optional[str] = None
    capture_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

@dataclass
class MonitorInfo:
    index: int
    left: int
    top: int
    width: int
    height: int

@dataclass
class CaptureMetadata:
    width: int
    height: int
    image_format: str
    file_size: int
    monitors: List[MonitorInfo] = field(default_factory=list)

@dataclass
class CapturedFrame:
    image_bytes: bytes
    metadata: CaptureMetadata
    capture_id: str
    timestamp: float

@dataclass
class PreparedCapture:
    """Image data ready to send to a vision provider."""
    primary_image_bytes: bytes
    primary_mime_type: str
    capture_id: str
    original_metadata: CaptureMetadata
    was_resized: bool = False
    resize_reason: str = ""
    tile_images: List[bytes] = field(default_factory=list)   # detail tiles
    tile_mime_types: List[str] = field(default_factory=list)
