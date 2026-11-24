from fastapi.testclient import TestClient

from app.database import init_db
from app.main import app

init_db()
client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_hardware():
    r = client.get("/api/v1/training/hardware")
    assert r.status_code == 200
    body = r.json()
    assert body["device"] in {"cpu", "cuda:0"}


def test_deploy_skeleton():
    r = client.get("/api/v1/deploy/status")
    assert r.status_code == 200
    assert r.json()["status"] == "skeleton"