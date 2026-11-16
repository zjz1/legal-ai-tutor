# -*- coding: utf-8 -*-
"""Provider 基类：把各家后端的差异收敛在实现内部，上层只面向统一契约。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..errors import ProviderAuthError, ProviderError, ProviderTimeout, ProviderUnavailable
from ..schemas import ChatMessage, ChatResult, Delta

_RETRYABLE_STATUS = (408, 409, 425, 429, 500, 502, 503, 504)


class BaseProvider(ABC):
    """所有模型提供方的统一接口。"""

    name: str = "base"

    def __init__(self, model: str, timeout: float = 25.0) -> None:
        self.model = model
        self.timeout = timeout

    @staticmethod
    def messages_to_dicts(messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.3,
        json_schema: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> ChatResult:
        """一次性返回完整结果。"""

    @abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.3,
        timeout: float | None = None,
    ) -> AsyncIterator[Delta]:
        """逐块返回增量。"""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(f"{self.name} 未实现向量化")

    async def health(self) -> dict[str, Any]:
        return {"ok": True, "provider": self.name, "model": self.model}

    async def aclose(self) -> None:
        return None


def raise_for_status(status: int, body: str, provider: str) -> None:
    """把 HTTP 状态翻译成带重试语义的项目异常。"""
    if status < 400:
        return
    snippet = (body or "")[:300]
    if status in (401, 403):
        raise ProviderAuthError(
            f"{provider} 鉴权失败（HTTP {status}）：{snippet}",
            provider=provider,
            status=status,
            hint="检查 API Key 是否有效、是否有该模型的调用权限",
        )
    error = ProviderError(f"{provider} 返回 HTTP {status}：{snippet}", provider=provider, status=status)
    error.retryable = status in _RETRYABLE_STATUS
    raise error


def map_transport_error(exc: Exception, provider: str) -> ProviderError:
    """把 httpx 传输层异常翻译成项目异常。"""
    if isinstance(exc, httpx.TimeoutException):
        return ProviderTimeout(f"{provider} 调用超时：{exc}", provider=provider)
    if isinstance(exc, (httpx.ConnectError, httpx.NetworkError, httpx.RemoteProtocolError)):
        return ProviderUnavailable(
            f"{provider} 连接失败：{exc}",
            provider=provider,
            hint="确认服务已启动且地址正确",
        )
    return ProviderError(f"{provider} 调用异常：{type(exc).__name__}: {exc}", provider=provider)


def estimate_tokens(text: str) -> int:
    """粗略估算 token：中文约 1 字 1 token，英文约 4 字符 1 token。"""
    return max(1, len(text) // 2)
