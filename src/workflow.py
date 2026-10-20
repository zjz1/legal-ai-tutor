# -*- coding: utf-8 -*-
"""legal-ai-tutor 双模式工作流 (Local Mode / Expert Mode)

本地模式：直接调用本地 Ollama 上的 legal-qwen 模型作答，数据不出本地。
专家模式：三层流水线
    第一层（本地）：抽取"法律主体 / 争议焦点 / 涉及法条"三要素
    第二层（云端）：把三要素交给云端大模型（OpenAI 兼容接口）生成专业分析
    第三层（本地）：把云端分析结果交回本地模型，做"法言法语"风格润色

配置全部通过环境变量注入，不硬编码任何密钥：
    OLLAMA_BASE_URL  默认 http://localhost:11434
    LOCAL_MODEL      默认 legal-qwen
    CLOUD_API_BASE   默认 https://api.deepseek.com/v1
    CLOUD_API_KEY    云端接口密钥（专家模式必需）
    CLOUD_MODEL      默认 deepseek-chat
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Callable, Dict, Optional

import requests

# --------------------------------------------------------------------------- 配置
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
LOCAL_MODEL: str = os.getenv("LOCAL_MODEL", "legal-qwen")
OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "300"))

CLOUD_API_BASE: str = os.getenv("CLOUD_API_BASE", "https://api.deepseek.com/v1").rstrip("/")
CLOUD_API_KEY: str = os.getenv("CLOUD_API_KEY", "")
CLOUD_MODEL: str = os.getenv("CLOUD_MODEL", "deepseek-chat")
CLOUD_TIMEOUT: int = int(os.getenv("CLOUD_TIMEOUT", "180"))

# --------------------------------------------------------------------------- 提示词
LOCAL_ANSWER_SYSTEM = (
    "你是一位执业多年的中国律师，精通民法、刑法、行政法。"
    "请用简洁、准确的语言回答用户的法律问题，必要时引用具体法条。"
)

EXTRACT_SYSTEM = (
    "你是一名法律文书分析助手，只负责信息抽取，不做法律分析。"
    "请严格按要求输出 JSON，不要输出任何解释性文字。"
)

EXTRACT_PROMPT = """请从下面的案情描述中抽取三要素，并且只输出如下 JSON 结构：

{{
  "legal_subjects": ["法律主体，例如：原告张某、被告某公司"],
  "dispute_focus": ["争议焦点，例如：借款合同是否成立并生效"],
  "legal_articles": ["涉及法条，例如：《民法典》第六百六十七条"]
}}

要求：
1. 只输出 JSON，不要加 Markdown 代码块标记，不要加任何前后缀说明；
2. 信息不足时对应字段填 "未提及"，不要编造当事人姓名或法条条号；
3. legal_articles 若无法确定具体条号，写法律名称即可。

案情描述：
{question}
"""

CLOUD_ANALYSIS_SYSTEM = (
    "你是一名资深法律专家，擅长基于争议焦点和法条出具专业的法律分析意见。"
    "请分点论述：① 法律关系认定；② 争议焦点逐项分析（结合法条）；③ 可行建议与风险提示。"
    "要求结论明确、条理清晰、不空泛。"
)

CLOUD_ANALYSIS_PROMPT = """请基于以下已抽取的案情要素，生成一份专业的法律分析意见。

【法律主体】{legal_subjects}
【争议焦点】{dispute_focus}
【涉及法条】{legal_articles}

【原始案情】
{question}
"""

POLISH_SYSTEM = (
    "你是一名法院文书的文字校对专家。"
    "请把下面这段草稿改写成规范的'法言法语'书面表达：使用规范法律术语、"
    "语句严谨、结构分点、去除口语化表达；不得改变原意，不得新增事实或结论。"
)

POLISH_PROMPT = """请对以下草稿进行法言法语风格润色，直接输出润色后的正文：

{draft}
"""


class WorkflowError(RuntimeError):
    """工作流执行异常。"""


# --------------------------------------------------------------------------- 底层调用
def ollama_chat(
    prompt: str,
    system: Optional[str] = None,
    model: str = LOCAL_MODEL,
    temperature: float = 0.3,
    json_mode: bool = False,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = OLLAMA_TIMEOUT,
) -> str:
    """调用本地 Ollama /api/chat 接口，返回 assistant 文本。"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        # 让 Ollama 走 JSON 约束解码，抽取层更稳定
        payload["format"] = "json"

    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise WorkflowError(
            f"无法连接本地 Ollama({base_url})，请确认 `ollama serve` 已启动且模型 {model} 已创建：{exc}"
        ) from exc

    if resp.status_code != 200:
        raise WorkflowError(f"Ollama 返回异常状态 {resp.status_code}: {resp.text[:200]}")

    content = (resp.json().get("message") or {}).get("content", "")
    return content.strip()


def cloud_chat(
    prompt: str,
    system: Optional[str] = None,
    api_base: str = CLOUD_API_BASE,
    api_key: str = CLOUD_API_KEY,
    model: str = CLOUD_MODEL,
    temperature: float = 0.4,
    timeout: int = CLOUD_TIMEOUT,
) -> str:
    """调用云端 OpenAI 兼容接口（DeepSeek / 通义 / Claude 中转等均可）。"""
    if not api_key:
        raise WorkflowError("未配置 CLOUD_API_KEY，专家模式的云端层不可用（可设置环境变量或在调用时注入自定义实现）。")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        resp = requests.post(
            f"{api_base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "temperature": temperature},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise WorkflowError(f"云端接口请求失败({api_base}): {exc}") from exc

    if resp.status_code != 200:
        raise WorkflowError(f"云端接口返回异常状态 {resp.status_code}: {resp.text[:200]}")

    choices = resp.json().get("choices") or []
    if not choices:
        raise WorkflowError("云端接口未返回任何 choices。")
    return (choices[0].get("message") or {}).get("content", "").strip()


# --------------------------------------------------------------------------- 模式一：本地模式
def local_answer(question: str, temperature: float = 0.3) -> str:
    """本地模式：本地 Ollama 直接作答。"""
    if not question or not question.strip():
        raise ValueError("question 不能为空。")
    return ollama_chat(
        prompt=question,
        system=LOCAL_ANSWER_SYSTEM,
        temperature=temperature,
    )


# --------------------------------------------------------------------------- 模式二：专家模式
def extract_elements(question: str, temperature: float = 0.1) -> Dict[str, Any]:
    """第一层（本地）：抽取法律主体、争议焦点、涉及法条。"""
    raw = ollama_chat(
        prompt=EXTRACT_PROMPT.format(question=question),
        system=EXTRACT_SYSTEM,
        temperature=temperature,
        json_mode=True,
    )

    data = _loads_json_loosely(raw)
    return {
        "legal_subjects": _as_list(data.get("legal_subjects")),
        "dispute_focus": _as_list(data.get("dispute_focus")),
        "legal_articles": _as_list(data.get("legal_articles")),
        "raw": raw,
    }


def expert_answer(
    question: str,
    cloud_llm: Optional[Callable[[str, str], str]] = None,
) -> Dict[str, Any]:
    """专家模式：本地要素抽取 -> 云端专业分析 -> 本地法言法语润色。

    cloud_llm: 可注入的云端调用实现，签名为 (prompt, system) -> str。
               不传则使用默认的 OpenAI 兼容 cloud_chat（需要 CLOUD_API_KEY）。
    """
    if not question or not question.strip():
        raise ValueError("question 不能为空。")

    # 1) 第一层：本地抽取结构化要素
    elements = extract_elements(question)

    # 2) 第二层：云端生成专业分析
    analysis_prompt = CLOUD_ANALYSIS_PROMPT.format(
        legal_subjects="；".join(elements["legal_subjects"]),
        dispute_focus="；".join(elements["dispute_focus"]),
        legal_articles="；".join(elements["legal_articles"]),
        question=question,
    )
    if cloud_llm is None:
        draft = cloud_chat(analysis_prompt, system=CLOUD_ANALYSIS_SYSTEM)
    else:
        draft = cloud_llm(analysis_prompt, CLOUD_ANALYSIS_SYSTEM)

    # 3) 第三层：本地法言法语润色
    final = ollama_chat(
        prompt=POLISH_PROMPT.format(draft=draft),
        system=POLISH_SYSTEM,
        temperature=0.2,
    )

    return {
        "question": question,
        "elements": elements,
        "cloud_draft": draft,
        "final_answer": final,
    }


# --------------------------------------------------------------------------- 工具函数
def _loads_json_loosely(text: str) -> Dict[str, Any]:
    """兼容模型偶尔带 Markdown 代码块或前后缀说明的输出。"""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise WorkflowError(f"要素抽取结果无法解析为 JSON：{text[:200]}")


def _as_list(value: Any) -> list:
    """把模型返回的 str / list / None 统一成 list[str]。"""
    if value is None:
        return ["未提及"]
    if isinstance(value, str):
        value = [value]
    if isinstance(value, list):
        items = [str(v).strip() for v in value if str(v).strip()]
        return items or ["未提及"]
    return ["未提及"]


# --------------------------------------------------------------------------- 测试
if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台中文输出
    except AttributeError:
        pass

    QUESTION = (
        "2023年5月，张三向李四借款20万元用于经营周转，双方口头约定月息2分，"
        "未签订书面合同。到期后张三拒不还款，李四多次催讨无果，"
        "现李四想起诉张三，请问该借款合同是否有效？李四能否主张利息？"
    )

    print("=" * 70)
    print("[本地模式] 直接调用 Ollama: " + LOCAL_MODEL)
    print("=" * 70)
    try:
        print(local_answer(QUESTION))
    except (WorkflowError, ValueError) as exc:
        print(f"本地模式测试失败：{exc}")

    print()
    print("=" * 70)
    print("[专家模式] 本地抽取 -> 云端分析 -> 本地润色")
    print("=" * 70)

    def _mock_cloud_llm(prompt: str, system: str) -> str:
        """无云端 Key 时的离线替身，仅用于验证流水线连通性。"""
        return (
            "一、法律关系认定：双方之间成立自然人间借款合同关系。\n"
            "二、争议焦点分析：自然人之间借款合同自提供借款时成立，口头约定不影响合同成立；"
            "未约定利息或约定不明的，视为不支付利息，但可主张逾期利息。\n"
            "三、建议：原告应固定转账凭证与催收记录，主张返还本金及逾期利息。"
        )

    try:
        if CLOUD_API_KEY:
            result = expert_answer(QUESTION)
        else:
            print("提示：未检测到 CLOUD_API_KEY，使用内置 mock 云端实现验证流水线。")
            result = expert_answer(QUESTION, cloud_llm=_mock_cloud_llm)

        print("--- 第一层：本地抽取的要素 ---")
        print(json.dumps(result["elements"], ensure_ascii=False, indent=2))
        print("--- 第二层：云端分析草稿 ---")
        print(result["cloud_draft"])
        print("--- 第三层：本地润色后的最终答复 ---")
        print(result["final_answer"])
    except (WorkflowError, ValueError) as exc:
        print(f"专家模式测试失败：{exc}")
