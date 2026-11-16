# -*- coding: utf-8 -*-
"""OpenAI 兼容 Provider（DeepSeek / 通义 / Kimi / vLLM 等）。"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..errors import ProviderAuthError
from ..schemas import ChatMessage, ChatResult, Delta, Usage
from .base import BaseProvider, map_transport_error, raise_for_status


class OpenAICompatProvider(BaseProvider):
    name = "openai"

    def __init__(self, api_base: str, api_key: str, model: str, timeout: float = 60.0) -> None:
        super().__init__(model=model, timeout=timeout)
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

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
            "temperature": temperature,
            "stream": stream,
        }
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "structured_output", "schema": json_schema, "strict": False},
            }
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
        if not self.api_key:
            raise ProviderAuthError("未配置云端 API Key", provider=self.name, hint="设置 CLOUD_API_KEY")

        started = time.perf_counter()
        payload = self._payload(messages, model, temperature, json_schema, stream=False)
        url = f"{self.api_base}/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=timeout or self.timeout) as client:
                resp = await client.post(url, json=payload, headers=self._headers)
                if resp.status_code == 400 and json_schema is not None:
                    # 部分兼容后端不支持 json_schema，降级为 json_object 再试一次
                    payload["response_format"] = {"type": "json_object"}
                    resp = await client.post(url, json=payload, headers=self._headers)
        except Exception as exc:
            raise map_transport_error(exc, self.name) from exc

        raise_for_status(resp.status_code, resp.text, self.name)
        data = resp.json()
        choices = data.get("choices") or [{}]
        text = (choices[0].get("message") or {}).get("content", "")
        raw_usage = data.get("usage") or {}
        return ChatResult(
            text=text.strip(),
            provider=self.name,
            model=data.get("model") or (model or self.model),
            usage=Usage(
                prompt_tokens=int(raw_usage.get("prompt_tokens") or 0),
                completion_tokens=int(raw_usage.get("completion_tokens") or 0),
                total_tokens=int(raw_usage.get("total_tokens") or 0),
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
        if not self.api_key:
            raise ProviderAuthError("未配置云端 API Key", provider=self.name)

        payload = self._payload(messages, model, temperature, None, stream=True)
        try:
            async with httpx.AsyncClient(timeout=timeout or self.timeout) as client:
                async with client.stream(
                    "POST", f"{self.api_base}/chat/completions", json=payload, headers=self._headers
                ) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", "ignore")
                        raise_for_status(resp.status_code, body, self.name)
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        chunk_text = line[5:].strip()
                        if chunk_text == "[DONE]":
                            yield Delta(done=True)
                            return
                        try:
                            chunk = json.loads(chunk_text)
                        except json.JSONDecodeError:
                            continue
                        choices = chunk.get("choices") or [{}]
                        piece = (choices[0].get("delta") or {}).get("content", "")
                        if piece:
                            yield Delta(text=piece)
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise map_transport_error(exc, self.name) from exc

    async def health(self) -> dict[str, Any]:
        return {
            "ok": bool(self.api_key),
            "provider": self.name,
            "model": self.model,
            "base": self.api_base,
            "detail": None if self.api_key else "未配置 CLOUD_API_KEY",
        }
