# -*- coding: utf-8 -*-
"""出云前的 PII 脱敏。

法律案情天然包含身份证号、手机号、银行卡、邮箱与当事人姓名。
云端只应看到占位符，返回后再按映射还原。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# (占位符前缀, 正则, 说明)
_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("身份证", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    ("手机号", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("银行卡", re.compile(r"(?<!\d)\d{16,19}(?!\d)")),
    ("邮箱", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
)

@dataclass
class RedactionResult:
    text: str
    mapping: dict[str, str] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(self.mapping)


class Redactor:
    """确定性脱敏器：同一原文重复出现时复用同一占位符。"""

    def redact(self, text: str, names: list[str] | None = None) -> RedactionResult:
        mapping: dict[str, str] = {}
        original_to_placeholder: dict[str, str] = {}
        out = text

        for label, pattern in _RULES:
            counter = 0
            pieces: list[str] = []
            last = 0
            for match in pattern.finditer(out):
                pieces.append(out[last : match.start()])
                value = match.group(0)
                if value in original_to_placeholder:
                    pieces.append(original_to_placeholder[value])
                else:
                    counter += 1
                    placeholder = f"[{label}{counter}]"
                    original_to_placeholder[value] = placeholder
                    mapping[placeholder] = value
                    pieces.append(placeholder)
                last = match.end()
            pieces.append(out[last:])
            out = "".join(pieces)

        for index, name in enumerate(self._collect_names(text, names)):
            placeholder = f"[当事人{self._party_label(index)}]"
            if name in original_to_placeholder:
                continue
            out = out.replace(name, placeholder)
            original_to_placeholder[name] = placeholder
            mapping[placeholder] = name

        return RedactionResult(text=out, mapping=mapping)

    @staticmethod
    def restore(text: str, mapping: dict[str, str]) -> str:
        """按占位符长度倒序还原，避免前缀互相覆盖。"""
        for placeholder in sorted(mapping, key=len, reverse=True):
            text = text.replace(placeholder, mapping[placeholder])
        return text

    @staticmethod
    def _party_label(index: int) -> str:
        return chr(ord("A") + index) if index < 26 else str(index + 1)

    @staticmethod
    def _collect_names(text: str, names: list[str] | None) -> list[str]:
        """显式传入的姓名优先；不传则不做姓名猜测，避免误伤常见词。"""
        if not names:
            return []
        found = [n for n in names if n and n in text]
        return sorted(set(found), key=len, reverse=True)


def redact(text: str, names: list[str] | None = None) -> RedactionResult:
    return Redactor().redact(text, names)


def restore(text: str, mapping: dict[str, str]) -> str:
    return Redactor.restore(text, mapping)
