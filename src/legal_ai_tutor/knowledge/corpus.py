# -*- coding: utf-8 -*-
"""法条语料：加载、规范化与查找。

法条以「一条一记录」存储，保留生效/废止日期，供引用校验判断时效性。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}

_ARTICLE_RE = re.compile(r"第([零〇一二两三四五六七八九十百千\d]+)条")
_BRACKETS = str.maketrans({"《": "", "》": "", " ": "", "　": ""})


def chinese_to_int(text: str) -> int | None:
    """把「六百六十七」或「667」转成整数。"""
    cleaned = text.strip()
    if not cleaned:
        return None
    if cleaned.isdigit():
        return int(cleaned)

    total = 0
    current = 0
    for char in cleaned:
        if char in _CN_DIGITS:
            current = _CN_DIGITS[char]
        elif char in _CN_UNITS:
            total += (current or 1) * _CN_UNITS[char]
            current = 0
        else:
            return None
    return total + current


def normalize_article(value: str) -> str | None:
    """统一成阿拉伯数字字符串，作为查找键。"""
    match = _ARTICLE_RE.search(value)
    raw = match.group(1) if match else value.strip()
    number = chinese_to_int(raw)
    return str(number) if number is not None else None


def normalize_law(name: str) -> str:
    return name.translate(_BRACKETS)


def law_aliases(name: str) -> set[str]:
    """「民法典」与「中华人民共和国民法典」视为同一部法。"""
    base = normalize_law(name)
    aliases = {base}
    prefix = "中华人民共和国"
    if base.startswith(prefix):
        aliases.add(base[len(prefix) :])
    else:
        aliases.add(prefix + base)
    return aliases


@dataclass(frozen=True)
class LawArticle:
    law: str
    article: str
    text: str
    effective_date: str | None = None
    repealed_date: str | None = None
    version: str | None = None

    @property
    def label(self) -> str:
        return f"《{self.law}》第{self.article}条"

    @property
    def is_current(self) -> bool:
        return not self.repealed_date

    def to_dict(self) -> dict[str, str | None]:
        return {
            "law": self.law,
            "article": self.article,
            "text": self.text,
            "effective_date": self.effective_date,
            "repealed_date": self.repealed_date,
            "version": self.version,
        }


class LawCorpus:
    def __init__(self, articles: list[LawArticle]) -> None:
        self.articles = list(articles)
        self._by_key: dict[tuple[str, str], LawArticle] = {}
        self._laws: set[str] = set()
        for item in self.articles:
            for alias in law_aliases(item.law):
                self._laws.add(alias)
                self._by_key[(alias, item.article)] = item

    @classmethod
    def load(cls, path: Path) -> LawCorpus:
        if not path.is_file():
            return cls([])
        articles: list[LawArticle] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            record = json.loads(line)
            article_no = normalize_article(str(record.get("article", "")))
            if not record.get("law") or not article_no or not record.get("text"):
                continue
            articles.append(
                LawArticle(
                    law=normalize_law(str(record["law"])),
                    article=article_no,
                    text=str(record["text"]),
                    effective_date=record.get("effective_date"),
                    repealed_date=record.get("repealed_date"),
                    version=record.get("version"),
                )
            )
        return cls(articles)

    def get(self, law: str, article: str) -> LawArticle | None:
        number = normalize_article(article)
        if number is None:
            return None
        for alias in law_aliases(law):
            found = self._by_key.get((alias, number))
            if found is not None:
                return found
        return None

    def has_law(self, law: str) -> bool:
        return any(alias in self._laws for alias in law_aliases(law))

    def __len__(self) -> int:
        return len(self.articles)
