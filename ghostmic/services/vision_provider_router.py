"""Vision provider router: selects the appropriate vision-capable model/provider."""
from __future__ import annotations

from typing import NamedTuple, Optional
from ghostmic.utils.logger import get_logger

logger = get_logger(__name__)

# Default vision models per provider
_GROQ_DEFAULT_VISION_MODEL = "qwen/qwen3.8-27b"
_OPENAI_DEFAULT_VISION_MODEL = "gpt-4o-mini"

# Models known to support image input (Groq)
_GROQ_VISION_CAPABLE_MODELS = frozenset({
    "qwen/qwen3.8-27b",
    "llava-v1.5-7b-4096-preview",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
})

# OpenAI models known to support vision
_OPENAI_VISION_CAPABLE_MODELS = frozenset({
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4-vision-preview",
})


class VisionProviderConfig(NamedTuple):
    provider: str          # "groq" | "openai"
    model: str             # model identifier
    api_key: str
    endpoint_url: Optional[str] = None   # None → use SDK default


class VisionProviderRouter:
    """Selects the appropriate vision provider and model for screen analysis.
    
    - Respects configured provider preferences.
    - Distinguishes vision-capable models from chat-only models.
    - Falls back to Groq vision when configured provider lacks vision support.
    """

    def resolve(self, ai_config: dict) -> VisionProviderConfig:
        """Resolve the vision provider config from the application AI config.
        
        Returns:
            VisionProviderConfig ready to use for a vision request.
            
        Raises:
            ValueError: If no usable vision provider is configured (missing API key).
        """
        expose_openai = bool(ai_config.get("expose_openai_provider", False))
        main_backend = str(ai_config.get("main_backend") or ai_config.get("backend", "groq")).strip().lower()
        
        # Prefer OpenAI vision when it is the main backend and configured
        if expose_openai and main_backend == "openai":
            openai_key = str(ai_config.get("openai_api_key", "")).strip()
            if openai_key:
                # Check if configured model supports vision
                configured_model = str(ai_config.get("openai_model", _OPENAI_DEFAULT_VISION_MODEL)).strip()
                vision_model = self._select_openai_vision_model(configured_model)
                logger.info("VisionProviderRouter: using OpenAI vision, model=%s", vision_model)
                return VisionProviderConfig(
                    provider="openai",
                    model=vision_model,
                    api_key=openai_key,
                )

        # Default: Groq vision
        groq_key = str(ai_config.get("groq_api_key", "")).strip()
        if not groq_key:
            raise ValueError(
                "No Groq API key configured. Add your Groq API key in Settings → AI "
                "before using screen analysis."
            )
        configured_model = str(
            ai_config.get("groq_vision_model", _GROQ_DEFAULT_VISION_MODEL)
        ).strip() or _GROQ_DEFAULT_VISION_MODEL
        vision_model = self._select_groq_vision_model(configured_model)
        logger.info("VisionProviderRouter: using Groq vision, model=%s", vision_model)
        return VisionProviderConfig(
            provider="groq",
            model=vision_model,
            api_key=groq_key,
            endpoint_url="https://api.groq.com/openai/v1",
        )

    def _select_groq_vision_model(self, configured: str) -> str:
        if configured in _GROQ_VISION_CAPABLE_MODELS:
            return configured
        logger.info(
            "VisionProviderRouter: configured Groq model %r may not support vision; "
            "using default %s",
            configured, _GROQ_DEFAULT_VISION_MODEL,
        )
        return _GROQ_DEFAULT_VISION_MODEL

    def _select_openai_vision_model(self, configured: str) -> str:
        if configured in _OPENAI_VISION_CAPABLE_MODELS:
            return configured
        logger.info(
            "VisionProviderRouter: configured OpenAI model %r may not support vision; "
            "using default %s",
            configured, _OPENAI_DEFAULT_VISION_MODEL,
        )
        return _OPENAI_DEFAULT_VISION_MODEL
