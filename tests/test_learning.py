# -*- coding: utf-8 -*-
"""学习模式：SM-2 间隔重复与三类教学产物。"""

from __future__ import annotations

import asyncio

from legal_ai_tutor.learning import LearningTutor, ReviewCard
from legal_ai_tutor.routing import Router


def test_sm2_first_three_reviews():
    card = ReviewCard(card_id="c1")
    card.review(5)
    assert card.interval_days == 1
    assert card.repetitions == 1

    card.review(5)
    assert card.interval_days == 6

    card.review(5)
    assert card.interval_days > 6


def test_sm2_failure_resets():
    card = ReviewCard(card_id="c1")
    card.review(5)
    card.review(5)
    card.review(1)
    assert card.repetitions == 0
    assert card.interval_days == 1
    assert card.ease >= 1.3


def test_sm2_roundtrip():
    card = ReviewCard(card_id="c1")
    card.review(4)
    restored = ReviewCard.from_dict(card.to_dict())
    assert restored.to_dict() == card.to_dict()


def test_breakdown_returns_structured(router: Router):
    tutor = LearningTutor(router)
    data = asyncio.run(tutor.breakdown("自然人之间的借款合同，自贷款人提供借款时成立。"))
    assert data["elements"]
    assert "example" in data


def test_quiz_returns_questions(router: Router):
    tutor = LearningTutor(router)
    questions = asyncio.run(tutor.quiz("自然人之间的借款合同，自贷款人提供借款时成立。", count=2))
    assert isinstance(questions, list)
    assert questions
    assert "stem" in questions[0]


def test_grade_returns_score(router: Router):
    tutor = LearningTutor(router)
    result = asyncio.run(tutor.grade("借款合同何时成立？", "提供借款时", "交付借款时成立"))
    assert 0 <= result["score"] <= 100
