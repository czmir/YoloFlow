from __future__ import annotations

import numpy as np


def generate_tiles(
    width: int,
    height: int,
    tile_size: int = 640,
    overlap: float = 0.2,
) -> list[tuple[int, int, int, int]]:
    step = max(1, int(tile_size * (1 - overlap)))
    tiles: list[tuple[int, int, int, int]] = []
    for y in range(0, height, step):
        for x in range(0, width, step):
            x2 = min(x + tile_size, width)
            y2 = min(y + tile_size, height)
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)
            tiles.append((x1, y1, x2, y2))
        if y + tile_size >= height:
            break
    # dedupe
    uniq = list(dict.fromkeys(tiles))
    return uniq


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> list[int]:
    """Simple greedy NMS for xyxy boxes."""
    if len(boxes) == 0:
        return []
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
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-6)
        order = rest[iou <= iou_threshold]
    return keep