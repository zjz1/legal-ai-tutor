# -*- coding: utf-8 -*-
"""运行时装配：把配置、Provider、检索器、工作流串起来。

被 API 与 CLI 共用，避免两处重复初始化逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .knowledge import HybridRetriever, LawCorpus
from .providers import build_cloud_provider, build_provider
from .routing import Router
from .workflow import LegalWorkflow


@dataclass
class Runtime:
    router: Router
    workflow: LegalWorkflow
    retriever: HybridRetriever | None
    corpus: LawCorpus


def build_runtime(settings: Settings) -> Runtime:
    local = build_provider(settings)

    # LLM_PROVIDER=openai 时，local 本身就是云端；专家模式无需再配一个云端
    if settings.provider == "openai":
        cloud = local
    else:
        cloud = build_cloud_provider(settings)

    router = Router(settings, local, cloud)
    corpus = LawCorpus.load(settings.law_dir / "sample_articles.jsonl")
    retriever = HybridRetriever(corpus) if len(corpus) else None
    workflow = LegalWorkflow(settings, router, retriever)
    return Runtime(router=router, workflow=workflow, retriever=retriever, corpus=corpus)
