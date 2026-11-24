from pathlib import Path

import numpy as np
import pytest

from app.services.tiling import class_aware_nms, generate_tiles, nms_boxes, remap_tile_box


def test_tiles_stay_within_bounds():
    width, height, tile = 4000, 3000, 640
    tiles = generate_tiles(width, height, tile_size=tile, overlap=0.2)
    assert tiles
    for x1, y1, x2, y2 in tiles:
        assert 0 <= x1 < x2 <= width
        assert 0 <= y1 < y2 <= height
        assert x2 - x1 <= tile
        assert y2 - y1 <= tile
    xs = {t[2] for t in tiles}
    ys = {t[3] for t in tiles}
    assert width in xs
    assert height in ys


def test_small_image_single_tile():
    tiles = generate_tiles(200, 100, tile_size=640, overlap=0.2)
    assert tiles == [(0, 0, 200, 100)]


def test_bad_tile_inputs():
    with pytest.raises(ValueError):
        generate_tiles(0, 10)
    with pytest.raises(ValueError):
        generate_tiles(10, 10, tile_size=0)
    with pytest.raises(ValueError):
        generate_tiles(10, 10, overlap=1.0)


def test_coord_remap():
    x1, y1, x2, y2 = remap_tile_box((10.0, 20.0, 30.0, 40.0), origin_x=100, origin_y=50)
    assert (x1, y1, x2, y2) == (110.0, 70.0, 130.0, 90.0)


def test_empty_detections_nms():
    assert nms_boxes(np.zeros((0, 4)), np.zeros((0,))) == []
    assert class_aware_nms([], [], []) == []


def test_class_aware_nms_keeps_other_class():
    boxes = [
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 1.0, 11.0, 11.0],
        [0.5, 0.5, 9.5, 9.5],
    ]
    scores = [0.9, 0.8, 0.95]
    classes = [0, 0, 1]
    keep = class_aware_nms(boxes, scores, classes, iou_threshold=0.5)
    kept_classes = {classes[i] for i in keep}
    assert kept_classes == {0, 1}
    assert len([i for i in keep if classes[i] == 0]) == 1


def test_nms_length_mismatch():
    with pytest.raises(ValueError):
        nms_boxes(np.zeros((2, 4)), np.zeros((1,)))


def test_infer_tiled_bad_image(tmp_path: Path):
    from app.services.inference_engine import infer_tiled

    missing = tmp_path / "nope.jpg"
    with pytest.raises(ValueError, match="cannot read image"):
        infer_tiled("yolov8n.pt", missing)
