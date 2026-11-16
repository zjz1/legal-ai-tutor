# -*- coding: utf-8 -*-
"""测试夹具：把 src 加入 sys.path，避免依赖 pip install -e。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_ai_tutor.config import load_settings  # noqa: E402
from legal_ai_tutor.knowledge import HybridRetriever, LawCorpus  # noqa: E402
from legal_ai_tutor.providers import MockProvider  # noqa: E402
from legal_ai_tutor.routing import Router  # noqa: E402
from legal_ai_tutor.workflow import LegalWorkflow  # noqa: E402

LAW_DIR = ROOT / "data" / "laws"
QUESTION = (
    "2023年5月，张三向李四借款20万元用于经营周转，双方口头约定月息2分，未签订书面合同。"
    "到期后张三拒不还款，李四多次催讨无果，请问该借款合同是否有效？李四能否主张利息？"
)


@pytest.fixture
def settings():
    return load_settings({"LLM_PROVIDER": "mock", "LAW_DIR": str(LAW_DIR)})


@pytest.fixture
def corpus(settings):
    return LawCorpus.load(settings.law_dir / "sample_articles.jsonl")


@pytest.fixture
def retriever(corpus):
    return HybridRetriever(corpus)


@pytest.fixture
def router(settings):
    return Router(settings, MockProvider(), None)


@pytest.fixture
def workflow(settings, router, retriever):
    return LegalWorkflow(settings, router, retriever)
