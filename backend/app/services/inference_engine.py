from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

from app.schemas import DetectionBox, InferenceResponse
from app.services.tiling import generate_tiles, nms_boxes


def _to_detections(result, names: dict[int, str]) -> list[DetectionBox]:
    boxes: list[DetectionBox] = []
    if result.boxes is None:
        return boxes
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        boxes.append(
            DetectionBox(
                class_id=cls_id,
                class_name=names.get(cls_id, str(cls_id)),
                confidence=conf,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )
    return boxes


def infer_pytorch(weights: str | Path, image_path: Path, conf: float = 0.25, device: str = "cpu") -> InferenceResponse:
    model = YOLO(str(weights))
    t0 = time.perf_counter()
    results = model.predict(source=str(image_path), conf=conf, device=device, verbose=False)
    latency = (time.perf_counter() - t0) * 1000
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    names = model.names
    dets = _to_detections(results[0], names)
    return InferenceResponse(
        backend="pytorch",
        latency_ms=latency,
        image_width=w,
        image_height=h,
        detections=dets,
    )


def infer_onnx(onnx_path: str | Path, image_path: Path, conf: float = 0.25) -> InferenceResponse:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    model = YOLO(str(onnx_path))  # ultralytics can load onnx for unified postprocess
    t0 = time.perf_counter()
    results = model.predict(source=str(image_path), conf=conf, verbose=False)
    latency = (time.perf_counter() - t0) * 1000
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    names = model.names
    dets = _to_detections(results[0], names)
    _ = session  # session retained for future raw ORT path
    return InferenceResponse(
        backend="onnx",
        latency_ms=latency,
        image_width=w,
        image_height=h,
        detections=dets,
    )


def infer_tiled_pytorch(
    weights: str | Path,
    image_path: Path,
    conf: float = 0.25,
    tile_size: int = 640,
    overlap: float = 0.2,
    device: str = "cpu",
) -> InferenceResponse:
    model = YOLO(str(weights))
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"cannot read image: {image_path}")
    h, w = img.shape[:2]
    tiles = generate_tiles(w, h, tile_size=tile_size, overlap=overlap)

    all_boxes: list[list[float]] = []
    all_scores: list[float] = []
    all_classes: list[int] = []

    t0 = time.perf_counter()
    for x1, y1, x2, y2 in tiles:
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        tmp = image_path.parent / f"_tile_{x1}_{y1}.jpg"
        cv2.imwrite(str(tmp), crop)
        try:
            results = model.predict(source=str(tmp), conf=conf, device=device, verbose=False)
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    bx = box.xyxy[0].tolist()
                    all_boxes.append([bx[0] + x1, bx[1] + y1, bx[2] + x1, bx[3] + y1])
                    all_scores.append(float(box.conf[0]))
                    all_classes.append(int(box.cls[0]))
        finally:
            if tmp.exists():
                tmp.unlink()

    latency = (time.perf_counter() - t0) * 1000
    names = model.names
    dets: list[DetectionBox] = []
    if all_boxes:
        keep = nms_boxes(np.array(all_boxes), np.array(all_scores))
        for idx in keep:
            cls_id = all_classes[idx]
            dets.append(
                DetectionBox(
                    class_id=cls_id,
                    class_name=names.get(cls_id, str(cls_id)),
                    confidence=all_scores[idx],
                    x1=all_boxes[idx][0],
                    y1=all_boxes[idx][1],
                    x2=all_boxes[idx][2],
                    y2=all_boxes[idx][3],
                )
            )

    return InferenceResponse(
        backend="pytorch-tiled",
        latency_ms=latency,
        image_width=w,
        image_height=h,
        detections=dets,
        tiled=True,
        tile_count=len(tiles),
    )


def compare_pytorch_onnx(
    weights: str | Path,
    onnx_path: str | Path,
    image_path: Path,
    conf: float = 0.25,
    device: str = "cpu",
) -> tuple[InferenceResponse, InferenceResponse, float, int]:
    pt = infer_pytorch(weights, image_path, conf=conf, device=device)
    ox = infer_onnx(onnx_path, image_path, conf=conf)
    # overlap metric: mean max IoU per pytorch box
    if not pt.detections:
        return pt, ox, 0.0, abs(len(pt.detections) - len(ox.detections))
    onnx_arr = np.array([[d.x1, d.y1, d.x2, d.y2] for d in ox.detections]) if ox.detections else np.zeros((0, 4))
    overlaps: list[float] = []
    for det in pt.detections:
        if onnx_arr.size == 0:
            overlaps.append(0.0)
            continue
        box = np.array([det.x1, det.y1, det.x2, det.y2])
        xx1 = np.maximum(box[0], onnx_arr[:, 0])
        yy1 = np.maximum(box[1], onnx_arr[:, 1])
        xx2 = np.minimum(box[2], onnx_arr[:, 2])
        yy2 = np.minimum(box[3], onnx_arr[:, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area = (box[2] - box[0]) * (box[3] - box[1])
        area_o = (onnx_arr[:, 2] - onnx_arr[:, 0]) * (onnx_arr[:, 3] - onnx_arr[:, 1])
        iou = inter / (area + area_o - inter + 1e-6)
        overlaps.append(float(iou.max()))
    mean_iou = float(np.mean(overlaps)) if overlaps else 0.0
    return pt, ox, mean_iou, len(pt.detections) - len(ox.detections)