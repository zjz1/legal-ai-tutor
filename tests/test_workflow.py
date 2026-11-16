# -*- coding: utf-8 -*-
"""工作流：本地模式、三层专家模式、SSE 事件序列。"""

from __future__ import annotations

import asyncio
import json

from legal_ai_tutor.providers import MockProvider
from legal_ai_tutor.providers.base import BaseProvider
from legal_ai_tutor.routing import Router
from legal_ai_tutor.schemas import ChatResult, Delta, Elements
from legal_ai_tutor.workflow import LegalWorkflow, _party_names

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


class SpyProvider(BaseProvider):
    """记录收到的完整 prompt，用于验证出云内容。"""

    name = "spy"

    def __init__(self, reply: str = "依据《中华人民共和国民法典》第六百七十九条，合同成立。") -> None:
        super().__init__(model="spy")
        self.reply = reply
        self.seen: list[str] = []

    async def chat(self, messages, *, model=None, temperature=0.3, json_schema=None, timeout=None) -> ChatResult:
        self.seen.append("\n".join(m.content for m in messages))
        return ChatResult(text=self.reply, provider=self.name, model=self.model)

    async def stream(self, messages, *, model=None, temperature=0.3, timeout=None):
        yield Delta(done=True)


def _workflow_with_spy(settings, retriever, spy: SpyProvider) -> LegalWorkflow:
    elements_json = json.dumps(
        {
            "legal_subjects": ["原告李四", "被告张三"],
            "dispute_focus": ["借款合同是否成立"],
            "legal_articles": [],
        },
        ensure_ascii=False,
    )
    local = MockProvider(responses={"案情描述": elements_json})
    return LegalWorkflow(settings, Router(settings, local, spy), retriever)


def test_party_names_extraction():
    names = _party_names(Elements(legal_subjects=["原告李四", "被告张三（示例）", "第三人"]))
    assert "李四" in names
    assert "张三" in names
    assert all(2 <= len(name) <= 4 for name in names)


def test_cloud_receives_redacted_prompt(settings, retriever):
    """出云前的 prompt 不得包含手机号、身份证号与当事人姓名。"""
    spy = SpyProvider()
    workflow = _workflow_with_spy(settings, retriever, spy)
    question = "原告李四称，被告张三电话13800138000，身份证110101199003071234，借款20万到期不还。"

    asyncio.run(workflow.expert_answer(question))

    sent = spy.seen[0]
    assert "13800138000" not in sent
    assert "110101199003071234" not in sent
    assert "李四" not in sent
    assert "[手机号" in sent
    assert "[当事人" in sent


def test_cloud_placeholders_are_restored(settings, retriever):
    """云端返回的占位符应在本地还原成真实姓名。"""
    spy = SpyProvider(reply="[当事人A]应返还借款本金。")
    workflow = _workflow_with_spy(settings, retriever, spy)

    result = asyncio.run(
        workflow.expert_answer("原告李四称，被告张三借款20万到期不还。")
    )
    assert "[当事人" not in result.draft
    assert "应返还借款本金" in result.draft


def test_redaction_can_be_disabled(settings):
    """REDACT_CLOUD=false 时原样发送，仅用于本地联调。"""
    from legal_ai_tutor.config import load_settings

    plain = load_settings({"LLM_PROVIDER": "mock", "REDACT_CLOUD": "false"})
    spy = SpyProvider()
    local = MockProvider()
    workflow = LegalWorkflow(plain, Router(plain, local, spy), None)

    asyncio.run(workflow.expert_answer("电话13800138000，借款不还。"))
    assert "13800138000" in spy.seen[0]
