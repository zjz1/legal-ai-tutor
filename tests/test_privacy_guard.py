# -*- coding: utf-8 -*-
"""隐私守卫：仓库源码与数据中不得出现真实个人信息或密钥。

测试夹具目录（tests/）使用虚构号码验证脱敏规则，故排除在扫描范围外。
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_PARTS = {
    ".git",
    "models",
    "datasets",
    "tests",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "reports",
    "node_modules",
}

TEXT_SUFFIXES = {".py", ".md", ".json", ".jsonl", ".yml", ".yaml", ".toml", ".txt", ".html", ".bat", ".ps1"}

PATTERNS = {
    "Windows 用户路径": re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE),
    "OpenAI 风格密钥": re.compile(r"\bsk-[A-Za-z0-9]{16,}"),
    "ModelScope Token": re.compile(r"\bms-[0-9a-f]{16,}"),
    "GitHub Token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    "手机号": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "身份证号": re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
}


def _candidate_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in SKIP_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != ".env.example":
            continue
        yield path, relative


def test_no_sensitive_data_in_repository():
    problems: list[str] = []
    for path, relative in _candidate_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                problems.append(f"{relative} -> {label}")

    assert not problems, "发现疑似敏感信息，请脱敏后再提交：" + "；".join(problems)
