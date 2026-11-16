# -*- coding: utf-8 -*-
"""工作流：本地模式、三层专家模式、SSE 事件序列。"""

from __future__ import annotations

import asyncio

from legal_ai_tutor.workflow import LegalWorkflow

QUESTION = (
    "2023年5月，张三向李四借款20万元用于经营周转，双方口头约定月息2分，未签订书面合同。"
    "到期后张三拒不还款，李四多次催讨无果，请问该借款合同是否有效？李四能否主张利息？"
)


def test_local_answer_returns_text(workflow: LegalWorkflow):
    result = asyncio.run(workflow.local_answer("口头借款合同是否成立？"))
    assert result.provider == "mock"
    assert result.answer
    assert result.elapsed_ms >= 0


def test_extract_elements_has_three_fields(workflow: LegalWorkflow):
    elements = asyncio.run(workflow.extract_elements(QUESTION))
    assert elements.legal_subjects
    assert elements.dispute_focus
    assert elements.legal_articles


def test_expert_degrades_when_cloud_missing(workflow: LegalWorkflow):
    result = asyncio.run(workflow.expert_answer(QUESTION))
    assert result.degraded is True
    assert result.draft
    assert result.final_answer
    assert result.elements.dispute_focus


def test_stream_expert_emits_expected_events(workflow: LegalWorkflow):
    async def collect():
        return [item async for item in workflow.stream_expert(QUESTION)]

    events = asyncio.run(collect())
    names = [item["event"] for item in events]
    assert "elements" in names
    assert "draft" in names
    assert "token" in names
    assert names[-1] == "done"


def test_extract_repairs_broken_json(tmp_path, settings):
    """模型返回脏 JSON 时应触发一次修复重试。"""
    from legal_ai_tutor.providers import MockProvider
    from legal_ai_tutor.routing import Router

    provider = MockProvider(responses={"案情描述": "这不是 JSON"})
    router = Router(settings, provider)
    workflow = LegalWorkflow(settings, router, None)

    # 修复请求里带有「请只输出符合要求的 JSON」，Mock 命中 schema 分支返回合法 JSON
    elements = asyncio.run(workflow.extract_elements("案情描述：甲借乙钱"))
    assert elements.legal_subjects
