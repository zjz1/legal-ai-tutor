# -*- coding: utf-8 -*-
"""法条引用抽取与校验。

模型引用的《X》第N条必须能在语料库中查到且未废止，否则标记为不可信。
"""

from __future__ import annotations

import re

from ..schemas import CitationCheck
from .corpus import LawCorpus, normalize_article

_CITE_RE = re.compile(r"《([^》]{2,40})》\s*第([零〇一二两三四五六七八九十百千\d]+)条")
_LAW_ONLY_RE = re.compile(r"《([^》]{2,40})》")


def extract_citations(text: str) -> list[tuple[str, str]]:
    """返回 [(法律名, 条号)]，按出现顺序去重。"""
    seen: set[tuple[str, str]] = set()
    results: list[tuple[str, str]] = []
    for match in _CITE_RE.finditer(text):
        law, raw_article = match.group(1).strip(), match.group(2)
        article = normalize_article(raw_article)
        if not article:
            continue
        key = (law, article)
        if key in seen:
            continue
        seen.add(key)
        results.append(key)
    return results


def validate_text(text: str, corpus: LawCorpus) -> list[CitationCheck]:
    """逐条校验引用，返回结构化结论。"""
    checks: list[CitationCheck] = []
    for law, article in extract_citations(text):
        found = corpus.get(law, article)
        if found is None:
            reason = "该法条未收录于本地法条库，无法核实" if corpus.has_law(law) else "法条库未收录该法律"
            checks.append(CitationCheck(law=law, article=article, valid=False, reason=reason))
            continue
        if not found.is_current:
            checks.append(
                CitationCheck(
                    law=found.law,
                    article=found.article,
                    valid=False,
                    reason=f"该条已于 {found.repealed_date} 废止",
                    effective_date=found.effective_date,
                    text=found.text,
                )
            )
            continue
        checks.append(
            CitationCheck(
                law=found.law,
                article=found.article,
                valid=True,
                reason="现行有效",
                effective_date=found.effective_date,
                text=found.text,
            )
        )
    return checks


def unknown_laws(text: str, corpus: LawCorpus) -> list[str]:
    """文中提到但库中没有的法律名。"""
    names = {m.group(1).strip() for m in _LAW_ONLY_RE.finditer(text)}
    return sorted(name for name in names if not corpus.has_law(name))


def annotate(text: str, corpus: LawCorpus) -> tuple[str, list[CitationCheck]]:
    """在正文末尾追加校验提示；无引用或全部有效时原样返回。"""
    checks = validate_text(text, corpus)
    invalid = [c for c in checks if not c.valid]
    if not invalid:
        return text, checks
    lines = [text, "", "【法条校验提示】以下引用未通过本地法条库校验，请以官方文本为准："]
    for item in invalid:
        lines.append(f"- 《{item.law}》第{item.article}条：{item.reason}")
    return "\n".join(lines), checks
