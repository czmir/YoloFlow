from app.services.infer_compare import (
    box_iou_xyxy,
    duplicate_report,
    match_detections_to_gt,
    pairwise_same_class_duplicates,
    parse_yolo_label_file,
    remap_boxes,
    yolo_norm_to_xyxy,
)
from app.services.tiling import class_aware_nms, generate_tiles
import numpy as np


def test_box_iou_identical():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    iou = box_iou_xyxy(a, a)
    assert iou.shape == (1, 1)
    assert abs(float(iou[0, 0]) - 1.0) < 1e-6


def test_box_iou_no_overlap():
    a = np.array([[0.0, 0.0, 1.0, 1.0]])
    b = np.array([[5.0, 5.0, 6.0, 6.0]])
    assert float(box_iou_xyxy(a, b)[0, 0]) == 0.0


def test_duplicate_pairs_same_class_only():
    boxes = [
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 1.0, 11.0, 11.0],
        [0.0, 0.0, 10.0, 10.0],
    ]
    classes = [0, 0, 1]
    pairs = pairwise_same_class_duplicates(boxes, classes, iou_threshold=0.5)
    assert len(pairs) == 1
    assert pairs[0][0] == 0 and pairs[0][1] == 1


def test_class_aware_nms_locked_against_cross_class():
    boxes = [
        [0.0, 0.0, 10.0, 10.0],
        [0.5, 0.5, 10.5, 10.5],
        [0.2, 0.2, 9.8, 9.8],
    ]
    scores = [0.9, 0.85, 0.95]
    classes = [0, 0, 1]
    keep = class_aware_nms(boxes, scores, classes, iou_threshold=0.5)
    kept_cls = sorted(classes[i] for i in keep)
    assert kept_cls == [0, 1]
    dup = duplicate_report(boxes, classes, [boxes[i] for i in keep], [classes[i] for i in keep])
    assert dup.pre_nms_duplicate_pairs >= 1
    assert dup.post_nms_duplicate_pairs == 0
    assert dup.suppressed == 1


def test_gt_match_precision_recall():
    gt_boxes = [[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]]
    gt_cls = [0, 1]
    det_boxes = [[0.5, 0.5, 10.5, 10.5], [200.0, 200.0, 210.0, 210.0]]
    det_cls = [0, 0]
    stats = match_detections_to_gt(det_boxes, det_cls, gt_boxes, gt_cls, iou_threshold=0.5)
    assert stats["tp"] == 1
    assert stats["fp"] == 1
    assert stats["fn"] == 1
    assert abs(stats["precision"] - 0.5) < 1e-9
    assert abs(stats["recall"] - 0.5) < 1e-9


def test_yolo_label_roundtrip():
    xyxy = yolo_norm_to_xyxy(0.5, 0.5, 0.2, 0.4, 100, 200)
    assert abs(xyxy[0] - 40.0) < 1e-6
    assert abs(xyxy[1] - 60.0) < 1e-6
    assert abs(xyxy[2] - 60.0) < 1e-6
    assert abs(xyxy[3] - 140.0) < 1e-6
    boxes, classes = parse_yolo_label_file("0 0.5 0.5 0.2 0.4\n1 0.1 0.1 0.1 0.1\n", 100, 100)
    assert classes == [0, 1]
    assert len(boxes) == 2


def test_remap_boxes_and_tile_count_for_mosaic():
    boxes = [[1.0, 2.0, 3.0, 4.0]]
    shifted = remap_boxes(boxes, 640, 0)
    assert shifted == [[641.0, 2.0, 643.0, 4.0]]
    tiles = generate_tiles(1280, 960, tile_size=320, overlap=0.2)
    assert len(tiles) > 1
    xs = {t[2] for t in tiles}
    ys = {t[3] for t in tiles}
    assert 1280 in xs
    assert 960 in ys


def test_empty_gt_all_false_positives():
    stats = match_detections_to_gt([[0, 0, 1, 1]], [0], [], [], iou_threshold=0.5)
    assert stats["fp"] == 1 and stats["tp"] == 0 and stats["fn"] == 0
