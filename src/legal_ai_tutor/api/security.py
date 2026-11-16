# -*- coding: utf-8 -*-
"""服务层横切能力：鉴权、限流、指标。"""

from __future__ import annotations

import hmac
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request


class RateLimiter:
    """进程内滑动窗口限流，单机部署足够；多实例时替换为 Redis。"""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        window = self._hits[key]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.per_minute:
            return False
        window.append(now)
        return True


def check_auth(request: Request, token: str) -> None:
    """静态 Bearer Token 鉴权；token 为空表示开发模式不校验。"""
    if not token:
        return
    header = request.headers.get("authorization", "")
    supplied = header[7:] if header.lower().startswith("bearer ") else ""
    if not hmac.compare_digest(supplied, token):
        raise HTTPException(
            status_code=401,
            detail={"code": "UNAUTHORIZED", "message": "缺少或错误的访问令牌"},
        )


class Metrics:
    """极简 Prometheus 文本格式指标，避免额外依赖。"""

    def __init__(self) -> None:
        self.started = time.time()
        self.counters: dict[tuple[str, int], int] = {}
        self.latency_sum: dict[str, int] = defaultdict(int)
        self.latency_count: dict[str, int] = defaultdict(int)
        self.provider_calls: dict[str, int] = defaultdict(int)

    def observe(self, route: str, status: int, elapsed_ms: int) -> None:
        self.counters[(route, status)] = self.counters.get((route, status), 0) + 1
        self.latency_sum[route] += elapsed_ms
        self.latency_count[route] += 1

    def count_provider(self, name: str) -> None:
        self.provider_calls[name] += 1

    def render(self) -> str:
        lines: list[str] = []
        for (route, status), value in sorted(self.counters.items()):
            lines.append(f'legal_ai_requests_total{{route="{route}",status="{status}"}} {value}')
        for route, total in sorted(self.latency_sum.items()):
            count = self.latency_count[route] or 1
            lines.append(f'legal_ai_request_avg_ms{{route="{route}"}} {total / count:.1f}')
        for name, value in sorted(self.provider_calls.items()):
            lines.append(f'legal_ai_provider_calls_total{{provider="{name}"}} {value}')
        lines.append(f"legal_ai_uptime_seconds {time.time() - self.started:.0f}")
        return "\n".join(lines) + "\n"
