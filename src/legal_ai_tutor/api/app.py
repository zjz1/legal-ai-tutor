# -*- coding: utf-8 -*-
"""FastAPI 应用工厂。

用工厂而非模块级 app，避免 import 时读取环境变量，便于测试注入配置。
"""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import PROJECT_ROOT, Settings, load_settings
from ..errors import LegalAIError
from ..learning import LearningTutor
from ..logging_setup import configure_logging, set_request_id
from ..runtime import build_runtime
from .security import Metrics, RateLimiter, check_auth

WEB_DIR = PROJECT_ROOT / "web"
_SKIP_LIMIT = ("/metrics", "/api/health")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="法律问题或案情描述")


class ExpertRequest(BaseModel):
    question: str = Field(..., min_length=1, description="法律问题或案情描述")


class ArticleRequest(BaseModel):
    law: str = Field(..., min_length=1)
    article: str = Field(..., min_length=1)


class QuizRequest(ArticleRequest):
    count: int = Field(3, ge=1, le=10)


class GradeRequest(BaseModel):
    question: str
    reference: str
    answer: str


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    configure_logging(settings.log_level)

    runtime = build_runtime(settings)
    tutor = LearningTutor(runtime.router)
    limiter = RateLimiter(settings.rate_limit_per_minute)
    metrics = Metrics()

    app = FastAPI(
        title="legal-ai-tutor",
        version="0.2.0",
        description="法律 AI 助手：本地模式 / 专家模式双工作流，法条 RAG 与引用校验",
    )
    app.state.settings = settings
    app.state.runtime = runtime
    app.state.metrics = metrics

    async def require_token(request: Request) -> None:
        check_auth(request, settings.api_token)

    def article_text(law: str, article: str) -> str:
        found = runtime.corpus.get(law, article)
        if found is None:
            raise HTTPException(status_code=404, detail={"code": "LAW_NOT_FOUND", "message": "本地法条库未收录该法条"})
        return found.text

    @app.middleware("http")
    async def observe(request: Request, call_next: Any) -> Any:
        rid = set_request_id(request.headers.get("x-request-id"))
        client = request.client.host if request.client else "unknown"
        if request.url.path not in _SKIP_LIMIT and not limiter.allow(client):
            return JSONResponse(
                status_code=429,
                content={"error": {"code": "RATE_LIMITED", "message": "请求过于频繁，请稍后再试"}},
            )
        started = time.perf_counter()
        response = await call_next(request)
        metrics.observe(request.url.path, response.status_code, int((time.perf_counter() - started) * 1000))
        response.headers["x-request-id"] = rid
        return response

    @app.exception_handler(LegalAIError)
    async def handle_legal_error(request: Request, exc: LegalAIError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content={"error": exc.to_dict()})

    # ------------------------------------------------------------------ 基础
    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        detail = await runtime.router.health()
        return {
            "status": "ok",
            "provider": settings.provider,
            "local_model": settings.local_model,
            "corpus_size": len(runtime.corpus),
            "retriever_ready": runtime.retriever is not None,
            "cloud_configured": settings.has_cloud,
            "concurrency": settings.max_concurrency,
            **detail,
        }

    @app.get("/metrics")
    async def prometheus_metrics() -> PlainTextResponse:
        return PlainTextResponse(metrics.render())

    # ------------------------------------------------------------------ 双模式
    @app.post("/api/local", dependencies=[Depends(require_token)])
    async def ask_local(req: AskRequest) -> dict[str, Any]:
        result = await runtime.workflow.local_answer(req.question)
        return result.model_dump()

    @app.post("/api/expert", dependencies=[Depends(require_token)])
    async def ask_expert(req: ExpertRequest) -> dict[str, Any]:
        result = await runtime.workflow.expert_answer(req.question)
        return result.model_dump()

    @app.post("/api/stream", dependencies=[Depends(require_token)])
    async def stream_expert(req: ExpertRequest) -> StreamingResponse:
        async def generator() -> Any:
            try:
                async for item in runtime.workflow.stream_expert(req.question):
                    data = json.dumps(item["data"], ensure_ascii=False)
                    yield f"event: {item['event']}\ndata: {data}\n\n"
            except LegalAIError as exc:
                payload = json.dumps({"error": exc.to_dict()}, ensure_ascii=False)
                yield f"event: error\ndata: {payload}\n\n"

        return StreamingResponse(
            generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------------ 法条
    @app.get("/api/laws/{law}/{article}")
    async def get_article(law: str, article: str) -> dict[str, Any]:
        found = runtime.corpus.get(law, article)
        if found is None:
            raise HTTPException(status_code=404, detail={"code": "LAW_NOT_FOUND", "message": "本地法条库未收录该法条"})
        return found.to_dict()

    @app.get("/api/laws")
    async def search_laws(q: str, top_k: int = 5) -> dict[str, Any]:
        if runtime.retriever is None:
            return {"items": [], "detail": "法条库为空"}
        retrieved = runtime.retriever.search(q, top_k=max(1, min(top_k, 20)))
        return {
            "items": [
                {**item.article.to_dict(), "score": round(item.score, 6), "source": item.source}
                for item in retrieved
            ]
        }

    # ------------------------------------------------------------------ 学习
    @app.post("/api/learning/breakdown", dependencies=[Depends(require_token)])
    async def learning_breakdown(req: ArticleRequest) -> dict[str, Any]:
        return await tutor.breakdown(article_text(req.law, req.article))

    @app.post("/api/learning/quiz", dependencies=[Depends(require_token)])
    async def learning_quiz(req: QuizRequest) -> dict[str, Any]:
        questions = await tutor.quiz(article_text(req.law, req.article), count=req.count)
        return {"questions": questions}

    @app.post("/api/learning/grade", dependencies=[Depends(require_token)])
    async def learning_grade(req: GradeRequest) -> dict[str, Any]:
        return await tutor.grade(req.question, req.reference, req.answer)

    # 前端静态页（最后挂载，避免抢占 /api）
    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

    return app

