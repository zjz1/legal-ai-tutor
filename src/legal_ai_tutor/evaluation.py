# -*- coding: utf-8 -*-
"""评估层：数据集加载、指标计算、带缓存的批量跑分。

指标口径：
- 法条精确率 / 召回率：预测引用的 (法律, 条号) 集合与金标集合对比
- 争议焦点召回：金标焦点是否被任一预测焦点包含（双向子串）
- 格式合规：抽取结果三个字段均非空
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .knowledge import extract_citations, normalize_law
from .workflow import LegalWorkflow


@dataclass
class EvalSample:
    id: str
    task: str
    question: str
    gold: dict[str, Any] = field(default_factory=dict)


def load_dataset(path: Path) -> list[EvalSample]:
    if not path.is_file():
        raise FileNotFoundError(f"评估集不存在：{path}")
    samples: list[EvalSample] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        record = json.loads(line)
        if not record.get("question"):
            continue
        samples.append(
            EvalSample(
                id=str(record.get("id") or f"sample-{number}"),
                task=str(record.get("task") or "extract"),
                question=str(record["question"]),
                gold=dict(record.get("gold") or {}),
            )
        )
    return samples


def normalize_refs(items: list[str]) -> set[tuple[str, str]]:
    """从「《民法典》第六百六十七条」这类文本中抽出 (法名, 条号)。"""
    refs: set[tuple[str, str]] = set()
    for item in items:
        for law, article in extract_citations(item):
            refs.add((normalize_law(law), article))
    return refs


def precision_recall(pred: set[tuple[str, str]], gold: set[tuple[str, str]]) -> tuple[float, float]:
    if not gold:
        return (1.0 if not pred else 0.0), 1.0
    if not pred:
        return 0.0, 0.0
    hits = len(pred & gold)
    return hits / len(pred), hits / len(gold)


def focus_recall(pred: list[str], gold: list[str]) -> float:
    if not gold:
        return 1.0
    hits = sum(1 for target in gold if any(target in item or item in target for item in pred))
    return hits / len(gold)


def format_compliance(elements: dict[str, list[str]]) -> float:
    fields = ("legal_subjects", "dispute_focus", "legal_articles")
    filled = sum(1 for key in fields if any(str(v).strip() for v in elements.get(key, [])))
    return filled / len(fields)


def _sample_key(sample: EvalSample, signature: str) -> str:
    raw = f"{signature}|{sample.task}|{sample.question}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class Evaluator:
    def __init__(self, workflow: LegalWorkflow, cache_path: Path | None = None) -> None:
        self.workflow = workflow
        self.cache_path = cache_path
        self.signature = f"{workflow.router.local.name}:{workflow.router.local.model}"
        self._cache: dict[str, dict[str, Any]] = self._load_cache()

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if self.cache_path and self.cache_path.is_file():
            try:
                return json.loads(self.cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8")

    async def _predict(self, sample: EvalSample) -> dict[str, Any]:
        key = _sample_key(sample, self.signature)
        if key in self._cache:
            return self._cache[key]

        if sample.task == "extract":
            elements = await self.workflow.extract_elements(sample.question)
            payload = {
                "elements": elements.model_dump(),
                "refs": [list(ref) for ref in sorted(normalize_refs(elements.legal_articles))],
            }
        else:
            result = await self.workflow.local_answer(sample.question)
            payload = {
                "answer": result.answer,
                "refs": [list(ref) for ref in sorted(normalize_refs([result.answer]))],
            }

        self._cache[key] = payload
        return payload

    async def run(self, samples: list[EvalSample], limit: int | None = None) -> dict[str, Any]:
        selected = samples[:limit] if limit else samples
        started = time.perf_counter()
        records: list[dict[str, Any]] = []

        for sample in selected:
            prediction = await self._predict(sample)
            pred_refs = {(law, article) for law, article in prediction.get("refs", [])}
            gold_refs = normalize_refs(sample.gold.get("articles", []))
            precision, recall = precision_recall(pred_refs, gold_refs)

            record: dict[str, Any] = {
                "id": sample.id,
                "task": sample.task,
                "citation_precision": round(precision, 4),
                "citation_recall": round(recall, 4),
            }
            if sample.task == "extract":
                elements = prediction.get("elements", {})
                record["focus_recall"] = round(
                    focus_recall(elements.get("dispute_focus", []), sample.gold.get("dispute_focus", [])), 4
                )
                record["format_compliance"] = round(format_compliance(elements), 4)
            records.append(record)

        self._save_cache()
        return self._aggregate(records, int((time.perf_counter() - started) * 1000))

    @staticmethod
    def _aggregate(records: list[dict[str, Any]], elapsed_ms: int) -> dict[str, Any]:
        def average(key: str) -> float | None:
            values = [r[key] for r in records if key in r]
            return round(sum(values) / len(values), 4) if values else None

        return {
            "config": "local" if records else "empty",
            "count": len(records),
            "citation_precision": average("citation_precision"),
            "citation_recall": average("citation_recall"),
            "focus_recall": average("focus_recall"),
            "format_compliance": average("format_compliance"),
            "elapsed_ms": elapsed_ms,
            "records": records,
        }


def format_report(report: dict[str, Any]) -> str:
    """把汇总结果渲染成对齐的文本表，便于 CLI 与 CI 日志查看。"""
    rows = [
        ("样本数", report.get("count")),
        ("法条精确率", report.get("citation_precision")),
        ("法条召回率", report.get("citation_recall")),
        ("争议焦点召回", report.get("focus_recall")),
        ("格式合规率", report.get("format_compliance")),
        ("耗时(ms)", report.get("elapsed_ms")),
    ]
    width = max(len(name) for name, _ in rows)
    lines = [f"{name.ljust(width)}  {value}" for name, value in rows]
    return "\n".join(lines)
