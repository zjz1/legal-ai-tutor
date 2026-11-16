# -*- coding: utf-8 -*-
"""双模式工作流。

本地模式：一次本地调用，数据不出本机。
专家模式：本地抽取要素 -> 检索候选法条 -> 云端分析 -> 本地润色 -> 引用校验。
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from pydantic import ValidationError

from .config import Settings
from .errors import SchemaError
from .knowledge import HybridRetriever, LawCorpus, annotate
from .prompts import ELEMENTS_SCHEMA, get_prompt
from .routing import Router
from .schemas import AnswerResult, ChatMessage, Elements, ExpertResult

logger = logging.getLogger(__name__)


def _format_elements(elements: Elements) -> str:
    return "\n".join(
        [
            "法律主体：" + "；".join(elements.legal_subjects or ["未提及"]),
            "争议焦点：" + "；".join(elements.dispute_focus or ["未提及"]),
            "涉及法条：" + "；".join(elements.legal_articles or ["未提及"]),
        ]
    )


class LegalWorkflow:
    def __init__(self, settings: Settings, router: Router, retriever: HybridRetriever | None = None) -> None:
        self.settings = settings
        self.router = router
        self.retriever = retriever
        self.corpus: LawCorpus = retriever.corpus if retriever is not None else LawCorpus([])

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _messages(system: str, user: str) -> list[ChatMessage]:
        return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]

    def _annotate(self, text: str) -> tuple[str, list[Any]]:
        if len(self.corpus) == 0:
            return text, []
        return annotate(text, self.corpus)

    def _retrieve_context(self, question: str, elements: Elements) -> str:
        if self.retriever is None or self.retriever.size == 0:
            return "（本地法条库为空，未提供候选条文；请勿臆造条号）"
        query = " ".join([question, *elements.dispute_focus, *elements.legal_articles])
        retrieved = self.retriever.search(query, top_k=8)
        return self.retriever.as_context(retrieved, limit=4) or "（未检索到相关条文）"

    @staticmethod
    def _parse_elements(text: str) -> Elements:
        raw = (text or "").strip()
        if not raw:
            raise SchemaError("要素抽取返回为空")
        try:
            return Elements.model_validate_json(raw)
        except ValidationError:
            pass

        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                return Elements.model_validate_json(raw[start : end + 1])
            except ValidationError as exc:
                raise SchemaError(f"要素抽取结果不符合 schema：{exc}") from exc
        raise SchemaError(f"要素抽取结果无法解析为 JSON：{raw[:120]}")

    # ------------------------------------------------------------------ 本地模式
    async def local_answer(self, question: str) -> AnswerResult:
        started = time.perf_counter()
        deadline = self.router.deadline()
        system, user = get_prompt("local_answer").render(question=question)
        result = await self.router.local_chat(self._messages(system, user), deadline=deadline)
        text, checks = self._annotate(result.text)
        return AnswerResult(
            answer=text,
            provider=result.provider,
            model=result.model,
            citations=checks,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    # ------------------------------------------------------------------ 专家模式
    async def extract_elements(self, question: str, deadline: float | None = None) -> Elements:
        deadline = deadline or self.router.deadline()
        system, user = get_prompt("extract").render(question=question)
        messages = self._messages(system, user)
        result = await self.router.local_chat(
            messages, json_schema=ELEMENTS_SCHEMA, temperature=0.1, deadline=deadline
        )
        try:
            return self._parse_elements(result.text)
        except SchemaError as first_error:
            logger.warning("要素抽取解析失败，触发一次修复：%s", first_error)
            repaired = messages + [
                ChatMessage(role="assistant", content=result.text),
                ChatMessage(role="user", content="上面的输出不是合法 JSON。请只输出符合要求的 JSON，不要任何解释或代码块标记。"),
            ]
            retry = await self.router.local_chat(
                repaired, json_schema=ELEMENTS_SCHEMA, temperature=0.0, deadline=deadline
            )
            return self._parse_elements(retry.text)

    async def expert_answer(self, question: str) -> ExpertResult:
        started = time.perf_counter()
        deadline = self.router.deadline()

        elements = await self.extract_elements(question, deadline)
        context = self._retrieve_context(question, elements)

        system, user = get_prompt("analyze").render(
            elements=_format_elements(elements), context=context, question=question
        )
        draft_result, degraded = await self.router.cloud_chat(self._messages(system, user), deadline=deadline)

        polish_system, polish_user = get_prompt("polish").render(draft=draft_result.text)
        final_result = await self.router.local_chat(
            self._messages(polish_system, polish_user), deadline=deadline
        )
        final_text, checks = self._annotate(final_result.text)

        return ExpertResult(
            question=question,
            elements=elements,
            draft=draft_result.text,
            final_answer=final_text,
            citations=checks,
            degraded=degraded,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    async def stream_expert(self, question: str) -> AsyncIterator[dict[str, Any]]:
        """SSE 事件流：stage / elements / draft / token / citations / done。"""
        started = time.perf_counter()
        deadline = self.router.deadline()

        yield {"event": "stage", "data": {"stage": "extract", "status": "start"}}
        elements = await self.extract_elements(question, deadline)
        yield {"event": "stage", "data": {"stage": "extract", "status": "done"}}
        yield {"event": "elements", "data": elements.model_dump()}

        yield {"event": "stage", "data": {"stage": "analyze", "status": "start"}}
        context = self._retrieve_context(question, elements)
        system, user = get_prompt("analyze").render(
            elements=_format_elements(elements), context=context, question=question
        )
        draft_result, degraded = await self.router.cloud_chat(self._messages(system, user), deadline=deadline)
        yield {"event": "stage", "data": {"stage": "analyze", "status": "done"}}
        yield {"event": "draft", "data": {"text": draft_result.text, "degraded": degraded}}

        yield {"event": "stage", "data": {"stage": "polish", "status": "start"}}
        polish_system, polish_user = get_prompt("polish").render(draft=draft_result.text)
        chunks: list[str] = []
        async for delta in self.router.stream_local(self._messages(polish_system, polish_user), deadline=deadline):
            if delta.text:
                chunks.append(delta.text)
                yield {"event": "token", "data": {"text": delta.text}}

        _, checks = self._annotate("".join(chunks))
        yield {"event": "citations", "data": [check.model_dump() for check in checks]}
        yield {
            "event": "done",
            "data": {"elapsed_ms": int((time.perf_counter() - started) * 1000), "degraded": degraded},
        }
