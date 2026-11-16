# -*- coding: utf-8 -*-
"""配置解析：默认值、非法值、布尔与路径。"""

from __future__ import annotations

import pytest

from legal_ai_tutor.config import load_settings
from legal_ai_tutor.errors import ConfigError


def test_default_provider_is_mock():
    settings = load_settings({})
    assert settings.provider == "mock"
    assert settings.max_concurrency == 1
    assert settings.has_cloud is False


def test_invalid_int_raises_readable_error():
    with pytest.raises(ConfigError) as exc:
        load_settings({"MAX_CONCURRENCY": "abc"})
    assert "MAX_CONCURRENCY" in str(exc.value)
    assert "整数" in str(exc.value)


def test_invalid_float_raises_readable_error():
    with pytest.raises(ConfigError) as exc:
        load_settings({"REQUEST_TIMEOUT": "abc"})
    assert "数字" in str(exc.value)


def test_invalid_bool_raises():
    with pytest.raises(ConfigError):
        load_settings({"ALLOW_CLIENT_KEY": "maybe"})


def test_unknown_provider_raises():
    with pytest.raises(ConfigError) as exc:
        load_settings({"LLM_PROVIDER": "llama"})
    assert "LLM_PROVIDER" in str(exc.value)


def test_cloud_key_marks_configured():
    settings = load_settings({"LLM_PROVIDER": "openai", "CLOUD_API_KEY": "sk-test"})
    assert settings.has_cloud is True


def test_relative_law_dir_is_resolved():
    settings = load_settings({"LAW_DIR": "data/laws"})
    assert settings.law_dir.is_absolute()
