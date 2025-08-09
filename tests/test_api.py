"""Pytest suite for RAG Medical Diagnosis API.

Tests focus on HTTP layer & contract while mocking heavy RAG / LLM logic.
We patch the imported symbols in app.main (get_answer / get_answer_with_debug)
to avoid loading sentence transformers or calling external Groq API.
"""

from fastapi.testclient import TestClient
import app.main as main_module
import pytest


@pytest.fixture(scope="module")
def client():
    return TestClient(main_module.app)


def test_root_endpoint(client):
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["name"].startswith("RAG Medical")
    assert "diagnose" in data


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert data["version"] == main_module.API_VERSION
    assert "timestamp" in data


def test_diagnose_success_no_debug(monkeypatch, client):
    stub_answer = "תשובה בדוקה"

    def fake_get_answer(q: str):  # noqa: D401
        assert isinstance(q, str)
        return stub_answer

    # Patch functions imported into main module
    monkeypatch.setattr(main_module, "get_answer", fake_get_answer)
    monkeypatch.setattr(main_module, "get_answer_with_debug", lambda q: (stub_answer, {"dbg": True}))

    r = client.post("/diagnose", json={"question": "יש לי כאב ראש"})
    assert r.status_code == 200
    data = r.json()
    assert data["question"] == "יש לי כאב ראש"
    assert data["answer"] == stub_answer
    # metadata basics
    meta = data.get("metadata")
    assert meta is not None
    assert meta["retrieved_conditions"] == 3
    assert 0 <= meta["confidence_score"] <= 1
    # debug should be absent when debug flag not set
    assert meta.get("debug") is None


def test_diagnose_success_with_debug(monkeypatch, client):
    stub_answer = "תשובה עם דיבאג"
    debug_info = {"embedder_model": "mock-model", "indices": [1, 2, 3]}

    monkeypatch.setattr(main_module, "get_answer", lambda q: stub_answer)
    monkeypatch.setattr(main_module, "get_answer_with_debug", lambda q: (stub_answer, debug_info))

    r = client.post("/diagnose?debug=true", json={"question": "חום וכאב גרון"})
    assert r.status_code == 200
    data = r.json()
    assert data["answer"] == stub_answer
    meta = data["metadata"]
    assert meta["debug"] == debug_info
    assert meta["response_time_ms"] >= 0


def test_diagnose_validation_error_missing_field(client):
    # Missing required body
    r = client.post("/diagnose", json={})
    assert r.status_code == 422
    # Response is from custom handler; ensure detail present
    detail = r.json()[0] if isinstance(r.json(), list) else r.json().get("detail")
    # We only assert structure loosely to avoid coupling to pydantic internals
    assert detail is not None


def test_diagnose_validation_error_empty_string(client):
    r = client.post("/diagnose", json={"question": ""})
    assert r.status_code == 422


def test_diagnose_internal_error(monkeypatch, client):
    def failing_get_answer(q: str):  # noqa: D401
        raise RuntimeError("Boom")

    monkeypatch.setattr(main_module, "get_answer", failing_get_answer)
    monkeypatch.setattr(main_module, "get_answer_with_debug", failing_get_answer)

    r = client.post("/diagnose", json={"question": "כאב בטן"})
    assert r.status_code == 500
    body = r.json()
    # Error shape: {"detail": {"detail": "Internal server error: ...", "error_code": "INTERNAL_ERROR", ...}}
    assert "detail" in body
    inner = body["detail"]
    assert inner.get("error_code") == "INTERNAL_ERROR"
    assert "Internal server error" in inner.get("detail", "")
