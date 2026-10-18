import httpx
import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture
def client():
    return TestClient(main.app)


def request_payload():
    return {
        "language": "python",
        "version": "3.12.0",
        "files": [
            {
                "name": "main.py",
                "content": "print('test')",
            }
        ],
    }


class FakeAsyncClient:
    """Mock the httpx client used by the execution endpoint."""

    def __init__(self, *, result=None, error=None, **kwargs):
        self.result = result
        self.error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def post(self, url, json):
        if self.error:
            raise self.error
        return self.result


def test_execute_success(client, monkeypatch):
    response = httpx.Response(
    200,
    json={
        "language": "python",
        "version": "3.12.0",
        "run": {"stdout": "test\n", "stderr": "", "code": 0},
    },
    request=httpx.Request(
        "POST",
        main.PISTON_EXECUTE_URL,
    ),
)

    monkeypatch.setattr(
        main.httpx,
        "AsyncClient",
        lambda **kwargs: FakeAsyncClient(result=response),
    )

    result = client.post("/execute", json=request_payload())

    assert result.status_code == 200
    assert result.json()["run"]["stdout"] == "test\n"


def test_execute_connection_failure_returns_503(client, monkeypatch):
    monkeypatch.setattr(
        main.httpx,
        "AsyncClient",
        lambda **kwargs: FakeAsyncClient(
            error=httpx.ConnectError("connection refused")
        ),
    )

    response = client.post("/execute", json=request_payload())

    assert response.status_code == 503
    assert response.json()["detail"] == "Local Piston service is unavailable"


def test_execute_timeout_returns_504(client, monkeypatch):
    monkeypatch.setattr(
        main.httpx,
        "AsyncClient",
        lambda **kwargs: FakeAsyncClient(
            error=httpx.ReadTimeout("request timed out")
        ),
    )

    response = client.post("/execute", json=request_payload())

    assert response.status_code == 504
    assert response.json()["detail"] == "Piston execution request timed out"


def test_execute_piston_http_error_returns_502(client, monkeypatch):
    # A real httpx.Response with a 4xx status and an attached request
    # raises httpx.HTTPStatusError from raise_for_status(), so the
    # endpoint's 502 branch is exercised without patching httpx.
    piston_response = httpx.Response(
        400,
        json={"message": "Unsupported runtime"},
        request=httpx.Request("POST", main.PISTON_EXECUTE_URL),
    )

    monkeypatch.setattr(
        main.httpx,
        "AsyncClient",
        lambda **kwargs: FakeAsyncClient(result=piston_response),
    )

    response = client.post("/execute", json=request_payload())

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail["message"] == "Piston rejected the execution request"
    assert detail["piston_status"] == 400
    assert detail["piston_error"] == "Unsupported runtime"
