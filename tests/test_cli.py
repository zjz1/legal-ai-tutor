# -*- coding: utf-8 -*-
"""CLI：Windows 控制台编码与命令冒烟。"""

from __future__ import annotations

import io
import sys

from legal_ai_tutor.cli import _force_utf8, main


def test_force_utf8_switches_cp1252_stream():
    """Windows runner 默认 cp1252，必须切到 utf-8 才能打印中文。"""
    original = sys.stdout
    fake = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    try:
        sys.stdout = fake
        assert sys.stdout.encoding.lower() == "cp1252"
        _force_utf8()
        assert sys.stdout.encoding.lower().startswith("utf-8")
    finally:
        sys.stdout = original


def test_index_command_returns_zero():
    assert main(["index"]) == 0
