# -*- coding: utf-8 -*-
"""评估：指标口径与跑分缓存。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from legal_ai_tutor.evaluation import (
    Evaluator,
    focus_recall,
    format_compliance,
    load_dataset,
    precision_recall,
)
from legal_ai_tutor.workflow import LegalWorkflow

ROOT = Path(__file__).resolve().parents[1]


def test_precision_recall_basic():
    pred = {("民法典", "667"), ("民法典", "679")}
    gold = {("民法典", "667")}
    precision, recall = precision_recall(pred, gold)
    assert precision == 0.5
    assert recall == 1.0


def test_precision_recall_empty_cases():
    assert precision_recall(set(), set()) == (1.0, 1.0)
    assert precision_recall(set(), {("民法典", "667")}) == (0.0, 0.0)
    assert precision_recall({("民法典", "667")}, set()) == (0.0, 1.0)


def test_focus_recall_uses_containment():
    assert focus_recall(["借款合同是否成立"], ["借款合同是否成立并生效"]) == 1.0
    assert focus_recall(["完全无关"], ["借款合同是否成立"]) == 0.0


def test_format_compliance():
    assert format_compliance({"legal_subjects": ["a"], "dispute_focus": ["b"], "legal_articles": ["c"]}) == 1.0
    assert format_compliance({"legal_subjects": ["a"], "dispute_focus": [], "legal_articles": []}) == 1 / 3


def test_evaluator_runs_and_caches(workflow: LegalWorkflow, tmp_path: Path):
    samples = load_dataset(ROOT / "eval" / "samples.jsonl")
    assert len(samples) >= 5

    cache = tmp_path / "cache.json"
    evaluator = Evaluator(workflow, cache_path=cache)
    report = asyncio.run(evaluator.run(samples, limit=2))

    assert report["count"] == 2
    assert report["citation_recall"] is not None
    assert cache.is_file()

    # 第二次应命中缓存，记录一致
    again = asyncio.run(Evaluator(workflow, cache_path=cache).run(samples, limit=2))
    assert again["records"] == report["records"]
