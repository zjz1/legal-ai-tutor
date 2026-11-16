# -*- coding: utf-8 -*-
"""日志配置：统一格式并注入请求 ID，便于串联一次请求的三层调用。"""

from __future__ import annotations

import contextvars
import logging
import sys
import uuid

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


def set_request_id(value: str | None = None) -> str:
    """设置当前上下文的请求 ID，返回最终值。"""
    rid = value or uuid.uuid4().hex[:12]
    _request_id.set(rid)
    return rid


def get_request_id() -> str:
    return _request_id.get()


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


def configure_logging(level: str = "INFO") -> None:
    """幂等配置根 logger。"""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if any(getattr(h, "_legal_ai", False) for h in root.handlers):
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s"))
    handler.addFilter(_RequestIdFilter())
    handler._legal_ai = True  # type: ignore[attr-defined]
    root.addHandler(handler)
