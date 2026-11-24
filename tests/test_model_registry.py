from pathlib import Path

from app.database import SessionLocal, init_db
from app.models import JobStatus, ModelVersion, TrainingJob
from app.services.inference_engine import export_detections_payload
from app.services.model_registry import register_job_model
from conftest import make_dataset

init_db()


def test_register_model_version(tmp_path: Path):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"pt-bytes")
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx-bytes")

    with SessionLocal() as db:
        ds = make_dataset(db, "registry")
        job = TrainingJob(
            name="reg-job",
            dataset_id=ds.id,
            status=JobStatus.succeeded.value,
            best_weights=str(ckpt),
            onnx_path=str(onnx),
            metrics_json={"metrics/mAP50(B)": 0.5},
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        mv = register_job_model(db, job)
        assert mv is not None
        assert mv.job_id == job.id
        assert mv.pytorch_sha256
        row = db.query(ModelVersion).filter(ModelVersion.job_id == job.id).one()
        assert row.onnx_path == str(onnx)


def test_export_detections_json():
    from app.schemas import DetectionBox

    boxes = [
        DetectionBox(class_id=0, class_name="a", confidence=0.9, x1=1, y1=2, x2=3, y2=4),
    ]
    media, name, body = export_detections_payload(boxes, "json")
    assert media == "application/json"
    assert "detections.json" == name
    assert "class_name" in body


def test_models_api_list(client=None):
    from fastapi.testclient import TestClient
    from app.main import app

    c = TestClient(app)
    r = c.get("/api/v1/models")
    assert r.status_code == 200
    assert isinstance(r.json(), list)