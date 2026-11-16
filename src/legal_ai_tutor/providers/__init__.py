# -*- coding: utf-8 -*-
"""Provider 注册表：按配置装配本地/云端实现。"""

from __future__ import annotations

from ..config import Settings
from ..errors import ConfigError
from .base import BaseProvider
from .mock import MockProvider
from .ollama import OllamaProvider
from .openai_compat import OpenAICompatProvider

__all__ = [
    "BaseProvider",
    "MockProvider",
    "OllamaProvider",
    "OpenAICompatProvider",
    "build_provider",
    "build_cloud_provider",
]


def build_provider(settings: Settings) -> BaseProvider:
    """按 LLM_PROVIDER 构建主提供方。"""
    if settings.provider == "mock":
        return MockProvider(model="mock-legal", timeout=settings.request_timeout)
    if settings.provider == "ollama":
        return OllamaProvider(
            base_url=settings.ollama_base_url,
            model=settings.local_model,
            timeout=settings.request_timeout,
        )
    if settings.provider == "openai":
        if not settings.cloud_api_key:
            raise ConfigError(
                "LLM_PROVIDER=openai 但未配置 CLOUD_API_KEY",
                hint="在 .env 里填写 CLOUD_API_KEY，或改用 LLM_PROVIDER=ollama/mock",
            )
        return OpenAICompatProvider(
            api_base=settings.cloud_api_base,
            api_key=settings.cloud_api_key,
            model=settings.cloud_model,
            timeout=settings.request_timeout,
        )
    raise ConfigError(f"未知的 LLM_PROVIDER：{settings.provider}")


def build_cloud_provider(settings: Settings) -> BaseProvider | None:
    """专家模式专用的云端提供方；未配置 Key 时返回 None，由编排层降级。"""
    if not settings.cloud_api_key:
        return None
    return OpenAICompatProvider(
        api_base=settings.cloud_api_base,
        api_key=settings.cloud_api_key,
        model=settings.cloud_model,
        timeout=settings.request_timeout,
    )
