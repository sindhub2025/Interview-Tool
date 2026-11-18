"""Screen capture service: captures the screen and returns a CapturedFrame."""
from __future__ import annotations

import time
from typing import Optional
from ghostmic.utils.logger import get_logger
from ghostmic.domain.captured_screen import (
    CaptureRequest, CaptureMetadata, CapturedFrame, MonitorInfo,
)

logger = get_logger(__name__)

class ScreenCaptureService:
    """Captures screenshots and returns CapturedFrame domain objects.
    
    Preserves the existing full-virtual-desktop capture behaviour as default.
    Does NOT perform image preprocessing or analysis.
    """

    def capture(self, request: Optional[CaptureRequest] = None) -> CapturedFrame:
        """Capture a screenshot and return a CapturedFrame.
        
        Args:
            request: Optional capture parameters. Defaults to full virtual desktop.
            
        Returns:
            CapturedFrame with raw image bytes and metadata.
        """
        if request is None:
            request = CaptureRequest()
        
        target = request.target
        if target in ("full_virtual_desktop", ""):
            return self._capture_full_virtual_desktop(request)
        elif target == "primary":
            return self._capture_primary_monitor(request)
        else:
            # Default to full virtual desktop for unknown targets
            logger.warning("Unknown capture target %r; falling back to full_virtual_desktop", target)
            return self._capture_full_virtual_desktop(request)

    def _capture_full_virtual_desktop(self, request: CaptureRequest) -> CapturedFrame:
        """Capture full virtual desktop as PNG bytes."""
        import mss
        from mss import tools as mss_tools

        with mss.mss() as sct:
            monitor = sct.monitors[0]  # monitor[0] = all monitors combined
            screenshot = sct.grab(monitor)
            png_bytes = mss_tools.to_png(screenshot.rgb, screenshot.size)
            
            # Collect per-monitor info
            monitors = []
            for idx, m in enumerate(sct.monitors[1:], start=1):
                monitors.append(MonitorInfo(
                    index=idx,
                    left=int(m.get("left", 0)),
                    top=int(m.get("top", 0)),
                    width=int(m.get("width", 0)),
                    height=int(m.get("height", 0)),
                ))

        metadata = CaptureMetadata(
            width=int(monitor["width"]),
            height=int(monitor["height"]),
            image_format="png",
            file_size=len(png_bytes),
            monitors=monitors,
        )
        logger.info(
            "ScreenCaptureService: captured full virtual desktop %dx%d, %d bytes",
            metadata.width, metadata.height, metadata.file_size,
        )
        return CapturedFrame(
            image_bytes=png_bytes,
            metadata=metadata,
            capture_id=request.capture_id,
            timestamp=request.timestamp,
        )

    def _capture_primary_monitor(self, request: CaptureRequest) -> CapturedFrame:
        """Capture primary monitor only (monitor index 1)."""
        import mss
        from mss import tools as mss_tools

        with mss.mss() as sct:
            if len(sct.monitors) < 2:
                # No separate primary — fall back
                return self._capture_full_virtual_desktop(request)
            monitor = sct.monitors[1]
            screenshot = sct.grab(monitor)
            png_bytes = mss_tools.to_png(screenshot.rgb, screenshot.size)

        metadata = CaptureMetadata(
            width=int(monitor["width"]),
            height=int(monitor["height"]),
            image_format="png",
            file_size=len(png_bytes),
            monitors=[MonitorInfo(index=1, left=int(monitor.get("left",0)),
                                   top=int(monitor.get("top",0)),
                                   width=int(monitor["width"]), height=int(monitor["height"]))],
        )
        return CapturedFrame(
            image_bytes=png_bytes,
            metadata=metadata,
            capture_id=request.capture_id,
            timestamp=request.timestamp,
        )
