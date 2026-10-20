# -*- coding: utf-8 -*-
"""legal-ai-tutor HTTP 服务层：把 workflow.py 的双模式能力暴露为 REST 接口 + 网页。

启动方式：
    python src\\server.py                 # 默认 http://127.0.0.1:8000
    uvicorn src.server:app --reload      # 开发热重载

接口一览：
    GET  /api/health   —— 健康检查（Ollama 是否在线、本地模型名、云端 Key 是否配置）
    POST /api/local    —— 本地模式：本地 Ollama 直接作答
    POST /api/expert   —— 专家模式：本地要素抽取 -> 云端分析 -> 本地法言法语润色
    GET  /             —— 前端页面 web/index.html
"""

from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# 允许 `python src/server.py` 直接运行（无需先安装为包）
sys.path.insert(0, str(Path(__file__).resolve().parent))

import workflow  # noqa: E402  (src/workflow.py)

WEB_DIR: Path = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(
    title="legal-ai-tutor",
    description="法律 AI 助手：本地模式 / 专家模式双工作流",
    version="0.1.0",
)


# --------------------------------------------------------------------------- 请求模型
class AskRequest(BaseModel):
    """本地模式请求体。"""

    question: str = Field(..., min_length=1, description="用户的法律问题或案情描述")
    temperature: float = Field(0.3, ge=0.0, le=1.5)


class ExpertRequest(BaseModel):
    """专家模式请求体。云端参数留空则回落到环境变量。"""

    question: str = Field(..., min_length=1, description="用户的法律问题或案情描述")
    cloud_api_key: Optional[str] = Field(None, description="云端接口密钥；留空则用环境变量 CLOUD_API_KEY")
    cloud_api_base: Optional[str] = Field(None, description="云端 OpenAI 兼容地址")
    cloud_model: Optional[str] = Field(None, description="云端模型名")
    use_mock_cloud: bool = Field(False, description="未配置云端 Key 时用内置 mock 走通三层流水线")


# --------------------------------------------------------------------------- 辅助
def _mock_cloud_llm(prompt: str, system: Optional[str] = None) -> str:
    """离线替身：无云端 Key 时验证「抽取 -> 分析 -> 润色」链路是否打通。"""
    return (
        "一、法律关系认定：双方之间成立民间借贷法律关系，属自然人之间的借款合同。\n"
        "二、争议焦点分析：\n"
        "  1. 借款合同是否成立生效——自然人之间借款合同自提供借款时成立，口头约定不影响成立；\n"
        "  2. 利息主张是否成立——约定不明的，视为不支付利息，但可主张逾期利息。\n"
        "三、可行建议与风险提示：固定转账凭证、催收记录等证据；主张返还本金及逾期利息；"
        "注意诉讼时效期间届满的风险。\n"
        "（说明：本条由内置 mock 生成，仅用于本地链路自测，不是真实云端输出。）"
    )


def _to_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, workflow.WorkflowError):
        return HTTPException(status_code=502, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


def _ollama_models() -> List[str]:
    """探测本地 Ollama 中已安装的模型列表。"""
    import requests

    resp = requests.get(f"{workflow.OLLAMA_BASE_URL}/api/tags", timeout=6)
    resp.raise_for_status()
    return [m.get("name", "") for m in resp.json().get("models", [])]


# --------------------------------------------------------------------------- 接口
@app.get("/api/health")
def health() -> Dict[str, Any]:
    """返回本地模型与云端配置的可用状态，供前端顶部状态条使用。"""
    try:
        models = _ollama_models()
        ollama_ok = True
    except Exception as exc:  # 连接失败 / 超时
        models, ollama_ok = [], False
    return {
        "ollama_ok": ollama_ok,
        "ollama_detail": None if ollama_ok else "无法连接 Ollama，请确认 ollama serve 已启动",
        "ollama_base_url": workflow.OLLAMA_BASE_URL,
        "local_model": workflow.LOCAL_MODEL,
        "local_model_ready": any(m.split(":")[0] == workflow.LOCAL_MODEL for m in models),
        "models": models,
        "cloud_key_configured": bool(workflow.CLOUD_API_KEY),
        "cloud_api_base": workflow.CLOUD_API_BASE,
        "cloud_model": workflow.CLOUD_MODEL,
    }


@app.post("/api/local")
def ask_local(req: AskRequest) -> Dict[str, Any]:
    """本地模式：数据不出本机，直接由 legal-qwen 作答。"""
    try:
        answer = workflow.local_answer(req.question, temperature=req.temperature)
    except Exception as exc:
        raise _to_http_error(exc)
    return {"mode": "local", "model": workflow.LOCAL_MODEL, "answer": answer}


@app.post("/api/expert")
def ask_expert(req: ExpertRequest) -> Dict[str, Any]:
    """专家模式：本地抽取要素 -> 云端专业分析 -> 本地法言法语润色。"""
    api_key = req.cloud_api_key or workflow.CLOUD_API_KEY
    cloud_llm = None

    if req.use_mock_cloud or not api_key:
        if not req.use_mock_cloud and not api_key:
            raise HTTPException(
                status_code=400,
                detail="专家模式需要云端密钥：请填写「云端 API Key」，或设置环境变量 CLOUD_API_KEY，"
                       "或勾选「使用内置 mock 云端」仅做链路自测。",
            )
        cloud_llm = _mock_cloud_llm
    else:
        cloud_llm = partial(
            workflow.cloud_chat,
            api_base=req.cloud_api_base or workflow.CLOUD_API_BASE,
            api_key=api_key,
            model=req.cloud_model or workflow.CLOUD_MODEL,
        )

    try:
        result = workflow.expert_answer(req.question, cloud_llm=cloud_llm)
    except Exception as exc:
        raise _to_http_error(exc)

    return {
        "mode": "expert",
        "model": workflow.LOCAL_MODEL,
        "cloud_model": "mock" if cloud_llm is _mock_cloud_llm else (req.cloud_model or workflow.CLOUD_MODEL),
        "elements": result["elements"],
        "cloud_draft": result["cloud_draft"],
        "final_answer": result["final_answer"],
    }


# 前端静态页（放在最后注册，避免抢占 /api 路由）
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    print(f"legal-ai-tutor 启动中 -> http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
