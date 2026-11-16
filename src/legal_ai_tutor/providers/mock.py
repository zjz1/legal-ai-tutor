# -*- coding: utf-8 -*-
"""离线 Mock 提供方。

作用：
1. 让整条流水线在没有模型服务时也能跑通（开发、CI、演示）；
2. 输出带明显标记，避免被误当成真实模型结果；
3. 结构化请求按 schema 返回可被 Pydantic 解析的 JSON。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import AsyncIterator
from typing import Any

from ..schemas import ChatMessage, ChatResult, Delta, Usage
from .base import BaseProvider, estimate_tokens

_MOCK_TAG = "【MOCK】"

_ELEMENTS_JSON = json.dumps(
    {
        "legal_subjects": ["原告（示例）", "被告（示例）"],
        "dispute_focus": ["合同是否成立并生效", "利息主张能否支持"],
        "legal_articles": ["《中华人民共和国民法典》第六百六十七条"],
    },
    ensure_ascii=False,
)

_QUIZ_JSON = json.dumps(
    {
        "questions": [
            {
                "type": "single_choice",
                "stem": "自然人间借款合同自何时成立？",
                "options": ["签字时", "口头约定时", "贷款人提供借款时", "催告时"],
                "answer": "C",
                "explanation": "依据《民法典》第六百七十九条，自贷款人提供借款时成立。",
            }
        ]
    },
    ensure_ascii=False,
)

_GRADE_JSON = json.dumps(
    {"score": 80, "hits": ["构成要件完整"], "misses": ["未提及举证责任"], "comment": "答案要点覆盖较好。"},
    ensure_ascii=False,
)

_ARTICLE_JSON = json.dumps(
    {
        "elements": ["双方之间存在借款合意", "贷款人实际提供借款"],
        "exceptions": ["自然人之间另有约定的，可不采用书面形式"],
        "burden": ["贷款人需证明款项已实际交付"],
        "example": "甲向乙转账并约定还款期，借款合同自转账时成立。",
    },
    ensure_ascii=False,
)


class MockProvider(BaseProvider):
    name = "mock"

    def __init__(
        self,
        model: str = "mock-legal",
        timeout: float = 25.0,
        responses: dict[str, str] | None = None,
        latency_ms: int = 0,
    ) -> None:
        super().__init__(model=model, timeout=timeout)
        self.responses = responses or {}
        self.latency_ms = latency_ms
        self.calls: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ 生成
    def _compose(self, messages: list[ChatMessage], json_schema: dict[str, Any] | None) -> str:
        prompt = messages[-1].content if messages else ""
        for key, value in self.responses.items():
            if key in prompt:
                return value

        if json_schema is not None:
            schema_text = json.dumps(json_schema, ensure_ascii=False)
            if "legal_subjects" in schema_text:
                return _ELEMENTS_JSON
            if "questions" in schema_text:
                return _QUIZ_JSON
            if "score" in schema_text:
                return _GRADE_JSON
            if "elements" in schema_text:
                return _ARTICLE_JSON
            return "{}"

        if "润色" in prompt:
            body = prompt.split("：", 1)[-1].strip()
            return f"{_MOCK_TAG}经法言法语润色后的表述如下：\n{body}"
        if "【法律主体】" in prompt:
            return (
                f"{_MOCK_TAG}一、法律关系认定：双方成立民间借贷法律关系。\n"
                "二、争议焦点分析：借款合同自提供借款时成立；利息约定不明视为不支付利息。\n"
                "三、可行建议与风险提示：固定转账凭证与催收记录，注意诉讼时效。"
            )
        if "出题" in prompt:
            return _QUIZ_JSON
        if "评分" in prompt:
            return _GRADE_JSON
        return (
            f"{_MOCK_TAG}当前为离线 Mock 输出，未调用任何模型服务。\n"
            f"收到的问题长度：{len(prompt)} 字。\n"
            "设置 LLM_PROVIDER=ollama 并启动 Ollama 后即可获得真实回答。"
        )

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.3,
        json_schema: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> ChatResult:
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000)
        text = self._compose(messages, json_schema)
        prompt_chars = sum(len(m.content) for m in messages)
        self.calls.append({"json": json_schema is not None, "chars": prompt_chars})
        return ChatResult(
            text=text,
            provider=self.name,
            model=model or self.model,
            usage=Usage(
                prompt_tokens=estimate_tokens("x" * prompt_chars),
                completion_tokens=estimate_tokens(text),
                total_tokens=estimate_tokens("x" * prompt_chars) + estimate_tokens(text),
            ),
            latency_ms=self.latency_ms,
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.3,
        timeout: float | None = None,
    ) -> AsyncIterator[Delta]:
        text = self._compose(messages, None)
        for index in range(0, len(text), 12):
            await asyncio.sleep(0)
            yield Delta(text=text[index : index + 12])
        yield Delta(done=True)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """确定性哈希向量：保证同样输入得到同样向量，便于离线测试检索。"""
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            raw = [(b - 127.5) / 127.5 for b in digest]
            norm = math.sqrt(sum(v * v for v in raw)) or 1.0
            vectors.append([v / norm for v in raw])
        return vectors

    async def health(self) -> dict[str, Any]:
        return {"ok": True, "provider": self.name, "model": self.model, "mode": "offline"}
