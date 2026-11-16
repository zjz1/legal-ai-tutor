# -*- coding: utf-8 -*-
"""HTTP 层：健康检查、双模式接口、法条查询、鉴权与指标。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from legal_ai_tutor.api.app import create_app
from legal_ai_tutor.config import load_settings

ROOT = Path(__file__).resolve().parents[1]
LAW_DIR = ROOT / "data" / "laws"
QUESTION = "张三向李四借款20万元，口头约定月息2分，到期不还，能否主张利息？"


def _client(**overrides: str) -> TestClient:
    env = {"LLM_PROVIDER": "mock", "LAW_DIR": str(LAW_DIR)}
    env.update(overrides)
    return TestClient(create_app(load_settings(env)))


@pytest.fixture
def client() -> TestClient:
    return _client()


def test_health_reports_mock(client: TestClient):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "mock"
    assert body["corpus_size"] > 0
    assert body["retriever_ready"] is True


def test_local_endpoint(client: TestClient):
    response = client.post("/api/local", json={"question": QUESTION})
    assert response.status_code == 200
    assert response.json()["answer"]


def test_expert_endpoint(client: TestClient):
    response = client.post("/api/expert", json={"question": QUESTION})
    assert response.status_code == 200
    body = response.json()
    assert body["elements"]["dispute_focus"]
    assert body["degraded"] is True


def test_law_lookup(client: TestClient):
    response = client.get("/api/laws/民法典/667")
    assert response.status_code == 200
    assert "借款合同" in response.json()["text"]


def test_law_lookup_missing(client: TestClient):
    response = client.get("/api/laws/民法典/9999")
    assert response.status_code == 404


def test_law_search(client: TestClient):
    response = client.get("/api/laws", params={"q": "借款 利息"})
    assert response.status_code == 200
    assert response.json()["items"]


def test_auth_required_when_token_configured():
    client = _client(API_TOKEN="secret")
    assert client.post("/api/local", json={"question": QUESTION}).status_code == 401
    ok = client.post("/api/local", json={"question": QUESTION}, headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200


def test_metrics_is_plain_text(client: TestClient):
    client.get("/api/health")
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "legal_ai_uptime_seconds" in response.text


def test_validation_error_on_empty_question(client: TestClient):
    response = client.post("/api/local", json={"question": ""})
    assert response.status_code == 422


def test_stream_endpoint_emits_sse(client: TestClient):
    with client.stream("POST", "/api/stream", json={"question": QUESTION}) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    assert "event: stage" in body
    assert "event: token" in body
    assert "event: done" in body
