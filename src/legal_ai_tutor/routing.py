# -*- coding: utf-8 -*-
"""模型路由：重试、熔断、降级、并发与时间预算。

上层（workflow）只调 local_chat / cloud_chat，不关心底层是哪家 Provider。
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .errors import BudgetExceeded, ProviderAuthError, ProviderError
from .providers.base import BaseProvider
from .schemas import ChatMessage, ChatResult, Delta

logger = logging.getLogger(__name__)


@dataclass
class BreakerState:
    failures: int = 0
    opened_at: float = 0.0

    def is_open(self, cooldown: float) -> bool:
        return self.opened_at > 0 and (time.monotonic() - self.opened_at) < cooldown


class CircuitBreaker:
    """连续失败达阈值后短路一段时间，避免把请求堆在坏掉的后端上。"""

    def __init__(self, threshold: int, cooldown: float) -> None:
        self.threshold = threshold
        self.cooldown = cooldown
        self._states: dict[str, BreakerState] = {}

    def allow(self, name: str) -> bool:
        state = self._states.setdefault(name, BreakerState())
        if state.opened_at and not state.is_open(self.cooldown):
            # 冷却结束，进入半开：放一个请求试探
            state.opened_at = 0.0
            state.failures = 0
        return not state.is_open(self.cooldown)

    def record_success(self, name: str) -> None:
        state = self._states.setdefault(name, BreakerState())
        state.failures = 0
        state.opened_at = 0.0

    def record_failure(self, name: str) -> None:
        state = self._states.setdefault(name, BreakerState())
        state.failures += 1
        if state.failures >= self.threshold:
            state.opened_at = time.monotonic()
            logger.warning("熔断触发：%s 连续失败 %d 次，冷却 %.0fs", name, state.failures, self.cooldown)


class Router:
    def __init__(self, settings: Settings, local: BaseProvider, cloud: BaseProvider | None = None) -> None:
        self.settings = settings
        self.local = local
        self.cloud = cloud
        self.breaker = CircuitBreaker(settings.circuit_failures, settings.circuit_cooldown)
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)

    # ------------------------------------------------------------------ 预算
    def deadline(self, seconds: float | None = None) -> float:
        return time.monotonic() + (seconds if seconds is not None else self.settings.total_deadline)

    @staticmethod
    def _remaining(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BudgetExceeded("请求已超出时间预算", hint="可调大 TOTAL_DEADLINE 或改用更小的模型")
        return remaining

    # ------------------------------------------------------------------ 单次调用
    async def _attempt(
        self,
        provider: BaseProvider,
        messages: list[ChatMessage],
        *,
        json_schema: dict[str, Any] | None,
        temperature: float,
        deadline: float | None,
    ) -> ChatResult:
        if not self.breaker.allow(provider.name):
            raise ProviderError(f"{provider.name} 处于熔断冷却期", provider=provider.name)

        timeout = self._remaining(deadline)
        timeout = min(timeout, self.settings.request_timeout) if timeout else self.settings.request_timeout

        async with self._semaphore:
            try:
                result = await provider.chat(
                    messages,
                    temperature=temperature,
                    json_schema=json_schema,
                    timeout=timeout,
                )
            except ProviderError as exc:
                if exc.retryable:
                    self.breaker.record_failure(provider.name)
                raise
        self.breaker.record_success(provider.name)
        return result

    async def _call_with_retry(
        self,
        provider: BaseProvider,
        messages: list[ChatMessage],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.3,
        deadline: float | None = None,
    ) -> ChatResult:
        last_error: ProviderError | None = None
        for attempt in range(self.settings.retries):
            try:
                return await self._attempt(
                    provider,
                    messages,
                    json_schema=json_schema,
                    temperature=temperature,
                    deadline=deadline,
                )
            except ProviderAuthError:
                raise
            except ProviderError as exc:
                last_error = exc
                if not exc.retryable or attempt == self.settings.retries - 1:
                    raise
                delay = min(4.0, 0.5 * (2**attempt)) + random.uniform(0, 0.2)
                logger.warning("%s 第 %d 次调用失败，%.1fs 后重试：%s", provider.name, attempt + 1, delay, exc)
                await asyncio.sleep(delay)
        raise last_error or ProviderError("调用失败", provider=provider.name)

    # ------------------------------------------------------------------ 对外接口
    async def local_chat(
        self,
        messages: list[ChatMessage],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.3,
        deadline: float | None = None,
    ) -> ChatResult:
        return await self._call_with_retry(
            self.local, messages, json_schema=json_schema, temperature=temperature, deadline=deadline
        )

    async def cloud_chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.4,
        deadline: float | None = None,
    ) -> tuple[ChatResult, bool]:
        """返回 (结果, 是否降级)。云端不可用时自动回落本地。"""
        if self.cloud is None:
            logger.info("未配置云端 Provider，专家模式降级为本地单模型")
            return await self.local_chat(messages, temperature=temperature, deadline=deadline), True

        try:
            result = await self._call_with_retry(self.cloud, messages, temperature=temperature, deadline=deadline)
            return result, False
        except ProviderError as exc:
            logger.warning("云端调用失败，降级本地：%s", exc)
            return await self.local_chat(messages, temperature=temperature, deadline=deadline), True

    async def stream_local(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.3,
        deadline: float | None = None,
    ) -> AsyncIterator[Delta]:
        if not self.breaker.allow(self.local.name):
            raise ProviderError(f"{self.local.name} 处于熔断冷却期", provider=self.local.name)
        timeout = self._remaining(deadline) or self.settings.request_timeout
        async with self._semaphore:
            async for delta in self.local.stream(messages, temperature=temperature, timeout=timeout):
                yield delta

    async def health(self) -> dict[str, Any]:
        local = await self.local.health()
        cloud = await self.cloud.health() if self.cloud else {"ok": False, "detail": "未配置云端 Key"}
        return {"local": local, "cloud": cloud, "concurrency": self.settings.max_concurrency}
