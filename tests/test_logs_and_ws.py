from pathlib import Path

from fastapi.testclient import TestClient

from app.database import SessionLocal, init_db
from app.main import app
from app.models import TrainingJob
from conftest import make_dataset

init_db()
client = TestClient(app)


def test_job_logs_tail():
    with SessionLocal() as db:
        ds = make_dataset(db, "logs")
        log_file = Path("/tmp/yoloflow-test-log.txt")
        log_file.write_text("line1\nline2\n", encoding="utf-8")
        job = TrainingJob(name="log-job", dataset_id=ds.id, log_path=str(log_file))
        db.add(job)
        db.commit()
        jid = job.id

    r = client.get(f"/api/v1/training/jobs/{jid}/logs?offset=0&limit=10")
    assert r.status_code == 200
    body = r.json()
    assert "line1" in body["content"]
    assert body["size"] >= 12


def test_resources_websocket_payload():
    with client.websocket_connect("/ws/resources") as ws:
        data = ws.receive_json()
        assert "cpu_percent" in data
        assert "queued_jobs" in data
        assert "paused_jobs" in data
        ws.send_text("ping")
        saw_pong = False
        for _ in range(6):
            msg = ws.receive_json()
            if msg.get("type") == "pong":
                saw_pong = True
                break
        assert saw_pong


def test_health():
    assert client.get("/health").json()["status"] == "ok"