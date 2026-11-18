"""Image preprocessor: validates and prepares captured images for vision APIs."""
from __future__ import annotations

import base64
import io
from typing import List, Optional, Tuple
from ghostmic.utils.logger import get_logger
from ghostmic.domain.captured_screen import CapturedFrame, PreparedCapture

logger = get_logger(__name__)

# Groq hard limit: 20 MB per image in the request body (base64 encoded adds ~33%)
# Keep the raw bytes well under this. We target 4MB raw ≈ ~5.3MB base64 to be safe.
GROQ_IMAGE_SIZE_LIMIT_BYTES: int = 20_000_000
# Target size for downscaling — keep readable text
_TARGET_MAX_BYTES: int = 4_000_000
# Minimum dimension we'll scale down to (to keep text readable)
_MIN_DIMENSION: int = 800
# JPEG quality for compressed output (high quality to preserve text legibility)
_JPEG_QUALITY: int = 92


class ImagePreprocessor:
    """Validates and prepares a CapturedFrame for vision API submission.
    
    Responsibilities:
    - Validates image dimensions and format
    - Handles oversized captures gracefully (resize with aspect ratio preserved)
    - Keeps readable text legible (does not aggressively downscale)
    - Returns PreparedCapture with original metadata preserved
    """

    def prepare(
        self,
        frame: CapturedFrame,
        *,
        provider_size_limit: int = GROQ_IMAGE_SIZE_LIMIT_BYTES,
    ) -> PreparedCapture:
        """Prepare a CapturedFrame for vision API submission.
        
        Returns PreparedCapture ready for the vision provider.
        """
        image_bytes = frame.image_bytes
        mime_type = "image/png"
        was_resized = False
        resize_reason = ""

        raw_size = len(image_bytes)
        logger.debug(
            "ImagePreprocessor: raw image %dx%d, %d bytes, format=%s",
            frame.metadata.width, frame.metadata.height,
            raw_size, frame.metadata.image_format,
        )

        if raw_size > provider_size_limit:
            logger.info(
                "ImagePreprocessor: image %d bytes exceeds provider limit %d; resizing",
                raw_size, provider_size_limit,
            )
            image_bytes, mime_type, resize_reason = self._resize_to_fit(
                image_bytes,
                frame.metadata.width,
                frame.metadata.height,
                target_max_bytes=min(_TARGET_MAX_BYTES, provider_size_limit // 4),
            )
            was_resized = True
            logger.info(
                "ImagePreprocessor: resized to %d bytes (%s); reason=%s",
                len(image_bytes), mime_type, resize_reason,
            )
        
        return PreparedCapture(
            primary_image_bytes=image_bytes,
            primary_mime_type=mime_type,
            capture_id=frame.capture_id,
            original_metadata=frame.metadata,
            was_resized=was_resized,
            resize_reason=resize_reason,
        )

    def _resize_to_fit(
        self,
        image_bytes: bytes,
        width: int,
        height: int,
        *,
        target_max_bytes: int,
    ) -> Tuple[bytes, str, str]:
        """Resize image to fit within target_max_bytes, preserving aspect ratio.
        
        Strategy:
        1. Try JPEG at high quality first (much smaller than PNG for photos/screenshots).
        2. If still too large, progressively scale down maintaining aspect ratio.
        3. Never scale below _MIN_DIMENSION on shortest side to keep text readable.
        
        Returns:
            (image_bytes, mime_type, resize_reason)
        """
        try:
            from PIL import Image  # type: ignore[import]
        except ImportError:
            # PIL not available — return original with warning
            logger.warning(
                "ImagePreprocessor: Pillow not installed; cannot resize image. "
                "Sending original (may exceed provider limit)."
            )
            return image_bytes, "image/png", "pillow_unavailable"

        img = Image.open(io.BytesIO(image_bytes))
        orig_w, orig_h = img.size
        
        # Step 1: Try JPEG conversion at high quality
        jpeg_buf = io.BytesIO()
        img_rgb = img.convert("RGB")
        img_rgb.save(jpeg_buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
        jpeg_bytes = jpeg_buf.getvalue()
        
        if len(jpeg_bytes) <= target_max_bytes:
            return jpeg_bytes, "image/jpeg", "jpeg_conversion"
        
        # Step 2: Scale down progressively
        scale = 0.85
        cur_w, cur_h = orig_w, orig_h
        result = jpeg_bytes  # fallback if loop doesn't execute
        new_w, new_h = cur_w, cur_h
        
        for _ in range(10):
            new_w = max(_MIN_DIMENSION, int(cur_w * scale))
            new_h = max(_MIN_DIMENSION, int(cur_h * scale))
            resized = img_rgb.resize((new_w, new_h), Image.LANCZOS)
            buf = io.BytesIO()
            resized.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
            result = buf.getvalue()
            if len(result) <= target_max_bytes:
                reason = f"scaled_to_{new_w}x{new_h}"
                return result, "image/jpeg", reason
            if new_w <= _MIN_DIMENSION and new_h <= _MIN_DIMENSION:
                break
            cur_w, cur_h = new_w, new_h
        
        # Last resort: return whatever we have (smallest scaled version)
        return result, "image/jpeg", f"best_effort_{new_w}x{new_h}"


def encode_image_data_url(image_bytes: bytes, mime_type: str = "image/png") -> str:
    """Return a data URL for a base64-encoded image payload."""
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
