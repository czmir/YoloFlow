from __future__ import annotations

import numpy as np


def generate_tiles(
    width: int,
    height: int,
    tile_size: int = 640,
    overlap: float = 0.2,
) -> list[tuple[int, int, int, int]]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")

    step = max(1, int(tile_size * (1 - overlap)))
    tiles: list[tuple[int, int, int, int]] = []
    y = 0
    while True:
        x = 0
        while True:
            x2 = min(x + tile_size, width)
            y2 = min(y + tile_size, height)
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)
            tiles.append((x1, y1, x2, y2))
            if x2 >= width:
                break
            x += step
        if y2 >= height:
            break
        y += step
    return list(dict.fromkeys(tiles))


def remap_tile_box(
    box_xyxy: tuple[float, float, float, float] | list[float],
    origin_x: int,
    origin_y: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    return x1 + origin_x, y1 + origin_y, x2 + origin_x, y2 + origin_y


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> list[int]:
    """Simple greedy NMS for xyxy boxes."""
    if boxes is None or len(boxes) == 0:
        return []
    if len(scores) != len(boxes):
        raise ValueError("boxes and scores length mismatch")
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = max(0.0, (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]))
        area_r = np.maximum(0.0, (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1]))
        union = area_i + area_r - inter + 1e-6
        iou = inter / union
        order = rest[iou <= iou_threshold]
    return keep


def class_aware_nms(
    boxes: list[list[float]],
    scores: list[float],
    classes: list[int],
    iou_threshold: float = 0.5,
) -> list[int]:
    """Keep overlapping boxes of different classes; suppress same-class IoU hits."""
    if not boxes:
        return []
    if not (len(boxes) == len(scores) == len(classes)):
        raise ValueError("boxes, scores, and classes must be the same length")
    keep: list[int] = []
    by_class: dict[int, list[int]] = {}
    for i, cls in enumerate(classes):
        by_class.setdefault(int(cls), []).append(i)
    for indices in by_class.values():
        arr = np.array([boxes[i] for i in indices], dtype=float)
        sc = np.array([scores[i] for i in indices], dtype=float)
        local_keep = nms_boxes(arr, sc, iou_threshold=iou_threshold)
        keep.extend(indices[i] for i in local_keep)
    return keep
