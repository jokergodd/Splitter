from __future__ import annotations

import logging
import uuid

from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_chat_service
from api.logging import REQUEST_ID_HEADER, generate_request_id


def _json_error(response):
    body = response.json()
    assert set(body) == {"code", "message", "details"}
    return body


def test_generate_request_id_looks_like_uuid():
    request_id = generate_request_id()

    assert str(uuid.UUID(request_id)) == request_id


def test_request_id_is_generated_and_logged_for_successful_requests(caplog):
    client = TestClient(app)

    with caplog.at_level(logging.INFO):
        response = client.get("/v1/health")

    request_id = response.headers[REQUEST_ID_HEADER]
    assert str(uuid.UUID(request_id)) == request_id

    request_logs = [
        record
        for record in caplog.records
        if record.name == "api.app" and getattr(record, "request_id", None) == request_id
    ]
    assert len(request_logs) == 1

    record = request_logs[0]
    assert record.method == "GET"
    assert record.path == "/v1/health"
    assert record.status_code == 200
    assert record.duration_ms >= 0


def test_request_id_is_propagated_on_error_responses():
    class FakeChatService:
        async def answer(self, *, question: str):
            raise ValueError("question is invalid")

    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    client = TestClient(app)
    request_id = "request-id-from-client"

    response = client.post(
        "/v1/chat/query",
        headers={REQUEST_ID_HEADER: request_id},
        json={"question": "hello"},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 400
    assert response.headers[REQUEST_ID_HEADER] == request_id
    assert _json_error(response) == {
        "code": "BAD_REQUEST",
        "message": "question is invalid",
        "details": {},
    }
