# -*- coding: utf-8 -*-
"""Provider 与路由：Mock 契约、重试、熔断、降级。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest

from legal_ai_tutor.config import load_settings
from legal_ai_tutor.errors import BudgetExceeded, ProviderError, ProviderUnavailable
from legal_ai_tutor.prompts import ELEMENTS_SCHEMA
from legal_ai_tutor.providers import MockProvider, build_provider
from legal_ai_tutor.providers.base import BaseProvider
from legal_ai_tutor.routing import Router
from legal_ai_tutor.schemas import ChatMessage, ChatResult, Delta, Elements


class FlakyProvider(BaseProvider):
    """前 N 次失败、之后成功，用于验证重试与熔断。"""

    name = "flaky"

    def __init__(self, failures: int = 1) -> None:
        super().__init__(model="flaky")
        self.failures = failures
        self.calls = 0

    async def chat(self, messages, *, model=None, temperature=0.3, json_schema=None, timeout=None) -> ChatResult:
        self.calls += 1
        if self.calls <= self.failures:
            raise ProviderUnavailable("故意失败", provider=self.name)
        return ChatResult(text="ok", provider=self.name, model=self.model)

    async def stream(self, messages, *, model=None, temperature=0.3, timeout=None) -> AsyncIterator[Delta]:
        yield Delta(done=True)


def _messages(text: str = "问题") -> list[ChatMessage]:
    return [ChatMessage(role="user", content=text)]


def test_mock_returns_structured_elements():
    provider = MockProvider()
    result = asyncio.run(provider.chat(_messages(), json_schema=ELEMENTS_SCHEMA))
    parsed = Elements.model_validate_json(result.text)
    assert parsed.legal_subjects
    assert parsed.dispute_focus


def test_mock_stream_yields_text_then_done():
    provider = MockProvider()

    async def collect() -> list[Delta]:
        return [delta async for delta in provider.stream(_messages("测试"))]

    chunks = asyncio.run(collect())
    assert any(delta.text for delta in chunks)
    assert chunks[-1].done is True


def test_mock_embed_is_deterministic_and_normalized():
    provider = MockProvider()
    first, second = asyncio.run(provider.embed(["借款合同", "借款合同"]))
    assert first == second
    assert abs(sum(v * v for v in first) - 1.0) < 1e-6


def test_build_provider_returns_mock(settings):
    assert build_provider(settings).name == "mock"


def test_router_retries_then_succeeds(settings):
    provider = FlakyProvider(failures=1)
    router = Router(settings, provider)
    result = asyncio.run(router.local_chat(_messages()))
    assert result.text == "ok"
    assert provider.calls == 2


def test_breaker_opens_after_threshold():
    settings = load_settings({"RETRIES": "1", "CIRCUIT_FAILURES": "2", "CIRCUIT_COOLDOWN": "60"})
    provider = FlakyProvider(failures=100)
    router = Router(settings, provider)

    for _ in range(2):
        with pytest.raises(ProviderError):
            asyncio.run(router.local_chat(_messages()))

    with pytest.raises(ProviderError) as exc:
        asyncio.run(router.local_chat(_messages()))
    assert "熔断" in str(exc.value)
    assert provider.calls == 2


def test_cloud_failure_falls_back_to_local():
    settings = load_settings({"RETRIES": "1"})
    router = Router(settings, MockProvider(), FlakyProvider(failures=100))
    result, degraded = asyncio.run(router.cloud_chat(_messages()))
    assert degraded is True
    assert result.provider == "mock"


def test_expired_deadline_raises(settings):
    router = Router(settings, MockProvider())
    with pytest.raises(BudgetExceeded):
        asyncio.run(router.local_chat(_messages(), deadline=time.monotonic() - 1))
