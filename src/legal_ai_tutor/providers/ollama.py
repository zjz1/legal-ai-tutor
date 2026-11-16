# -*- coding: utf-8 -*-
"""本地 Ollama Provider（/api/chat）。"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..schemas import ChatMessage, ChatResult, Delta, Usage
from .base import BaseProvider, map_transport_error, raise_for_status


class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float = 25.0) -> None:
        super().__init__(model=model, timeout=timeout)
        self.base_url = base_url.rstrip("/")

    def _payload(
        self,
        messages: list[ChatMessage],
        model: str | None,
        temperature: float,
        json_schema: dict[str, Any] | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": self.messages_to_dicts(messages),
            "stream": stream,
            "options": {"temperature": temperature},
        }
        if json_schema is not None:
            # 新版 Ollama 支持直接传 JSON Schema 做约束解码
            payload["format"] = json_schema
        return payload

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.3,
        json_schema: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> ChatResult:
        started = time.perf_counter()
        payload = self._payload(messages, model, temperature, json_schema, stream=False)
        try:
            async with httpx.AsyncClient(timeout=timeout or self.timeout) as client:
                resp = await client.post(f"{self.base_url}/api/chat", json=payload)
        except Exception as exc:  # 传输层异常
            raise map_transport_error(exc, self.name) from exc

        raise_for_status(resp.status_code, resp.text, self.name)
        data = resp.json()
        text = (data.get("message") or {}).get("content", "")
        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        completion_tokens = int(data.get("eval_count") or 0)
        return ChatResult(
            text=text.strip(),
            provider=self.name,
            model=data.get("model") or (model or self.model),
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.3,
        timeout: float | None = None,
    ) -> AsyncIterator[Delta]:
        payload = self._payload(messages, model, temperature, None, stream=True)
        try:
            async with httpx.AsyncClient(timeout=timeout or self.timeout) as client:
                async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", "ignore")
                        raise_for_status(resp.status_code, body, self.name)
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        chunk = json.loads(line)
                        piece = (chunk.get("message") or {}).get("content", "")
                        if piece:
                            yield Delta(text=piece)
                        if chunk.get("done"):
                            yield Delta(done=True)
                            return
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise map_transport_error(exc, self.name) from exc

    async def health(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            models = [m.get("name", "") for m in resp.json().get("models", [])]
        except Exception as exc:
            return {"ok": False, "provider": self.name, "model": self.model, "detail": f"{type(exc).__name__}: {exc}"}

        ready = any(name.split(":")[0] == self.model for name in models)
        return {"ok": True, "provider": self.name, "model": self.model, "model_ready": ready, "models": models}
