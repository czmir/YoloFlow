"""Full-image vs tiled inference comparison helpers (no model I/O)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix for xyxy boxes. a: (N,4), b: (M,4) -> (N,M)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    ax1 = a[:, 0][:, None]
    ay1 = a[:, 1][:, None]
    ax2 = a[:, 2][:, None]
    ay2 = a[:, 3][:, None]
    xx1 = np.maximum(ax1, b[:, 0])
    yy1 = np.maximum(ay1, b[:, 1])
    xx2 = np.minimum(ax2, b[:, 2])
    yy2 = np.minimum(ay2, b[:, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_a = np.maximum(0.0, (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = np.maximum(0.0, (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))
    union = area_a + area_b - inter + 1e-6
    return inter / union


def pairwise_same_class_duplicates(
    boxes: list[list[float]] | np.ndarray,
    classes: list[int] | np.ndarray,
    iou_threshold: float = 0.5,
) -> list[tuple[int, int, float]]:
    """Same-class pairs with IoU above threshold (i < j). Used as a duplicate-box score."""
    if len(boxes) == 0:
        return []
    arr = np.asarray(boxes, dtype=float).reshape(-1, 4)
    cls = np.asarray(classes, dtype=int).reshape(-1)
    if len(arr) != len(cls):
        raise ValueError("boxes and classes length mismatch")
    ious = box_iou_xyxy(arr, arr)
    pairs: list[tuple[int, int, float]] = []
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if cls[i] != cls[j]:
                continue
            iou = float(ious[i, j])
            if iou > iou_threshold:
                pairs.append((i, j, iou))
    return pairs


def yolo_norm_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> list[float]:
    w = float(image_width)
    h = float(image_height)
    bw = width * w
    bh = height * h
    cx = x_center * w
    cy = y_center * h
    return [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]


def parse_yolo_label_file(text: str, image_width: int, image_height: int) -> tuple[list[list[float]], list[int]]:
    boxes: list[list[float]] = []
    classes: list[int] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        boxes.append(yolo_norm_to_xyxy(xc, yc, bw, bh, image_width, image_height))
        classes.append(cls)
    return boxes, classes


def match_detections_to_gt(
    det_boxes: list[list[float]] | np.ndarray,
    det_classes: list[int] | np.ndarray,
    gt_boxes: list[list[float]] | np.ndarray,
    gt_classes: list[int] | np.ndarray,
    iou_threshold: float = 0.5,
) -> dict[str, float | int]:
    """Greedy class-aware matching at a single IoU threshold."""
    dets = np.asarray(det_boxes, dtype=float).reshape(-1, 4) if len(det_boxes) else np.zeros((0, 4))
    dcls = np.asarray(det_classes, dtype=int).reshape(-1) if len(det_classes) else np.zeros((0,), dtype=int)
    gts = np.asarray(gt_boxes, dtype=float).reshape(-1, 4) if len(gt_boxes) else np.zeros((0, 4))
    gcls = np.asarray(gt_classes, dtype=int).reshape(-1) if len(gt_classes) else np.zeros((0,), dtype=int)
    n_det = len(dets)
    n_gt = len(gts)
    if n_gt == 0:
        return {
            "tp": 0,
            "fp": n_det,
            "fn": 0,
            "precision": 0.0 if n_det else 1.0,
            "recall": 1.0,
            "n_det": n_det,
            "n_gt": 0,
        }
    if n_det == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": n_gt,
            "precision": 0.0,
            "recall": 0.0,
            "n_det": 0,
            "n_gt": n_gt,
        }
    ious = box_iou_xyxy(dets, gts)
    gt_used = np.zeros(n_gt, dtype=bool)
    tp = 0
    for di in range(n_det):
        best_j = -1
        best_iou = iou_threshold
        for gj in range(n_gt):
            if gt_used[gj] or dcls[di] != gcls[gj]:
                continue
            iou = float(ious[di, gj])
            if iou > best_iou:
                best_iou = iou
                best_j = gj
        if best_j >= 0:
            gt_used[best_j] = True
            tp += 1
    fp = n_det - tp
    fn = n_gt - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "n_det": int(n_det),
        "n_gt": int(n_gt),
    }


def remap_boxes(boxes: list[list[float]], dx: float, dy: float) -> list[list[float]]:
    out: list[list[float]] = []
    for x1, y1, x2, y2 in boxes:
        out.append([x1 + dx, y1 + dy, x2 + dx, y2 + dy])
    return out


@dataclass
class DuplicateReport:
    pre_nms_count: int
    post_nms_count: int
    pre_nms_duplicate_pairs: int
    post_nms_duplicate_pairs: int
    suppressed: int
    iou_threshold: float

    def as_dict(self) -> dict:
        return asdict(self)


def duplicate_report(
    pre_boxes: list[list[float]],
    pre_classes: list[int],
    post_boxes: list[list[float]],
    post_classes: list[int],
    iou_threshold: float = 0.5,
) -> DuplicateReport:
    pre_pairs = pairwise_same_class_duplicates(pre_boxes, pre_classes, iou_threshold)
    post_pairs = pairwise_same_class_duplicates(post_boxes, post_classes, iou_threshold)
    return DuplicateReport(
        pre_nms_count=len(pre_boxes),
        post_nms_count=len(post_boxes),
        pre_nms_duplicate_pairs=len(pre_pairs),
        post_nms_duplicate_pairs=len(post_pairs),
        suppressed=max(0, len(pre_boxes) - len(post_boxes)),
        iou_threshold=iou_threshold,
    )
