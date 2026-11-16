# -*- coding: utf-8 -*-
"""脱敏：结构化 PII 与姓名占位、还原。"""

from __future__ import annotations

from legal_ai_tutor.privacy import Redactor, redact, restore


def test_structured_pii_is_replaced():
    text = "张三电话13800138000，身份证110101199003071234，邮箱a.b@example.com。"
    result = redact(text)
    assert "13800138000" not in result.text
    assert "110101199003071234" not in result.text
    assert "a.b@example.com" not in result.text
    assert "[手机号1]" in result.text
    assert "[身份证1]" in result.text


def test_restore_roundtrip():
    text = "张三电话13800138000。"
    result = redact(text, names=["张三"])
    restored = restore(result.text, result.mapping)
    assert restored == text


def test_repeated_value_reuses_placeholder():
    text = "13800138000 与 13800138000 是同一号码"
    result = redact(text)
    assert result.text.count("[手机号1]") == 2
    assert len(result.mapping) == 1


def test_no_pii_returns_unchanged():
    text = "借款合同自贷款人提供借款时成立。"
    result = redact(text)
    assert result.changed is False
    assert result.text == text


def test_names_without_dict_are_not_guessed():
    """不做姓名猜测，避免误伤「合同」「法院」等常用词。"""
    result = Redactor().redact("张三向李四借款")
    assert result.text == "张三向李四借款"
