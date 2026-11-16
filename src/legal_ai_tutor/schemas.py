# -*- coding: utf-8 -*-
"""跨层数据契约：所有 Provider / 工作流 / API 共用同一套模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: Role
    content: str


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResult(BaseModel):
    text: str
    provider: str
    model: str
    usage: Usage = Field(default_factory=Usage)
    latency_ms: int = 0


class Delta(BaseModel):
    """流式增量。"""

    text: str = ""
    done: bool = False


class Elements(BaseModel):
    """第一层抽取结果。"""

    legal_subjects: list[str] = Field(default_factory=list)
    dispute_focus: list[str] = Field(default_factory=list)
    legal_articles: list[str] = Field(default_factory=list)


class CitationCheck(BaseModel):
    """单条法条引用的校验结论。"""

    law: str
    article: str
    valid: bool
    reason: str
    effective_date: str | None = None
    text: str | None = None


class AnswerResult(BaseModel):
    answer: str
    provider: str
    model: str
    degraded: bool = False
    citations: list[CitationCheck] = Field(default_factory=list)
    elapsed_ms: int = 0


class ExpertResult(BaseModel):
    question: str
    elements: Elements
    draft: str
    final_answer: str
    citations: list[CitationCheck] = Field(default_factory=list)
    degraded: bool = False
    elapsed_ms: int = 0
