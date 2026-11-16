# -*- coding: utf-8 -*-
"""学习模式：法条拆解、自测题、作答评分，以及 SM-2 间隔重复。

设计上复用同一套 Router / Provider，因此离线 Mock 也能完整演示。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from .errors import SchemaError
from .prompts import ARTICLE_SCHEMA, GRADE_SCHEMA, QUIZ_SCHEMA, get_prompt
from .routing import Router
from .schemas import ChatMessage


@dataclass
class ReviewCard:
    """SM-2 记忆卡状态。"""

    card_id: str
    ease: float = 2.5
    interval_days: int = 0
    repetitions: int = 0
    due: date = field(default_factory=date.today)

    def review(self, quality: int) -> ReviewCard:
        """quality 0-5：低于 3 视为遗忘，重新开始。"""
        if not 0 <= quality <= 5:
            raise ValueError("quality 必须在 0-5 之间")

        if quality < 3:
            self.repetitions = 0
            self.interval_days = 1
        else:
            self.repetitions += 1
            if self.repetitions == 1:
                self.interval_days = 1
            elif self.repetitions == 2:
                self.interval_days = 6
            else:
                self.interval_days = max(1, round(self.interval_days * self.ease))

        self.ease = max(1.3, self.ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
        self.due = date.today() + timedelta(days=self.interval_days)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "ease": round(self.ease, 3),
            "interval_days": self.interval_days,
            "repetitions": self.repetitions,
            "due": self.due.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewCard:
        return cls(
            card_id=str(data["card_id"]),
            ease=float(data.get("ease", 2.5)),
            interval_days=int(data.get("interval_days", 0)),
            repetitions=int(data.get("repetitions", 0)),
            due=date.fromisoformat(data["due"]) if data.get("due") else date.today(),
        )


def _parse_json(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise SchemaError(f"模型未返回 JSON：{raw[:120]}")
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise SchemaError(f"JSON 解析失败：{exc}") from exc
    if not isinstance(data, dict):
        raise SchemaError("模型返回的 JSON 不是对象")
    return data


class LearningTutor:
    def __init__(self, router: Router) -> None:
        self.router = router

    async def breakdown(self, article_text: str) -> dict[str, Any]:
        """法条拆解：构成要件 / 例外 / 举证责任 / 示例。"""
        system, user = get_prompt("learning_article").render(article_text=article_text)
        result = await self.router.local_chat(
            [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)],
            json_schema=ARTICLE_SCHEMA,
            temperature=0.2,
        )
        data = _parse_json(result.text)
        return {
            "elements": list(data.get("elements") or []),
            "exceptions": list(data.get("exceptions") or []),
            "burden": list(data.get("burden") or []),
            "example": str(data.get("example") or ""),
        }

    async def quiz(self, article_text: str, count: int = 3) -> list[dict[str, Any]]:
        """生成自测题。"""
        system, user = get_prompt("learning_quiz").render(article_text=article_text, count=count)
        result = await self.router.local_chat(
            [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)],
            json_schema=QUIZ_SCHEMA,
            temperature=0.6,
        )
        data = _parse_json(result.text)
        questions = data.get("questions")
        if not isinstance(questions, list):
            raise SchemaError("自测题缺少 questions 字段")
        return questions[:count]

    async def grade(self, question: str, reference: str, answer: str) -> dict[str, Any]:
        """按要点评分。"""
        system, user = get_prompt("learning_grade").render(question=question, reference=reference, answer=answer)
        result = await self.router.local_chat(
            [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)],
            json_schema=GRADE_SCHEMA,
            temperature=0.1,
        )
        data = _parse_json(result.text)
        try:
            score = int(data.get("score", 0))
        except (TypeError, ValueError) as exc:
            raise SchemaError("评分不是整数") from exc
        return {
            "score": max(0, min(100, score)),
            "hits": list(data.get("hits") or []),
            "misses": list(data.get("misses") or []),
            "comment": str(data.get("comment") or ""),
        }
