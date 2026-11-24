from pathlib import Path

from app.services.dataset_validator import (
    discover_yaml_in_tree,
    scan_yolo_folder,
    validate_yolo_dataset,
    _label_path_for_image,
)


def test_validate_missing_yaml(tmp_path: Path):
    stats, validation = validate_yolo_dataset(tmp_path / "missing.yaml")
    assert not validation.valid
    assert stats.num_images_train == 0


def test_validate_minimal_dataset(tmp_path: Path):
    root = tmp_path / "ds"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    img = root / "images" / "train" / "a.jpg"
    img.write_bytes(
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfe\x84\xb4\x93\xff\xd9"
    )
    (root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    yaml = root / "data.yaml"
    yaml.write_text(f"path: {root}\ntrain: images/train\nval: images/train\nnames: [obj]\n", encoding="utf-8")

    stats, validation = validate_yolo_dataset(yaml)
    assert validation.valid
    assert stats.num_images_train == 1
    assert stats.num_labels_train == 1


def test_class_id_out_of_range(tmp_path: Path):
    root = tmp_path / "ds"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    img = root / "images" / "train" / "a.jpg"
    img.write_bytes(
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfe\x84\xb4\x93\xff\xd9"
    )
    (root / "labels" / "train" / "a.txt").write_text("5 0.5 0.5 0.2 0.2\n")
    yaml = root / "data.yaml"
    yaml.write_text(f"path: {root}\ntrain: images/train\nval: images/train\nnames: [only]\n", encoding="utf-8")

    _, validation = validate_yolo_dataset(yaml)
    assert not validation.valid
    codes = {i.code for i in validation.issues}
    assert "class_id_out_of_range" in codes


def test_orphan_label_warning(tmp_path: Path):
    root = tmp_path / "ds"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    img = root / "images" / "train" / "a.jpg"
    img.write_bytes(
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfe\x84\xb4\x93\xff\xd9"
    )
    (root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (root / "labels" / "train" / "orphan.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    yaml = root / "data.yaml"
    yaml.write_text(f"path: {root}\ntrain: images/train\nval: images/train\nnames: [obj]\n", encoding="utf-8")

    _, validation = validate_yolo_dataset(yaml)
    assert validation.valid
    assert any(i.code == "orphan_label" for i in validation.issues)


def test_label_path_mapping():
    p = Path("/data/coco/images/train2017/x.jpg")
    lbl = _label_path_for_image(p)
    assert lbl == Path("/data/coco/labels/train2017/x.txt")


def test_discover_yaml(tmp_path: Path):
    nested = tmp_path / "imported" / "coco"
    nested.mkdir(parents=True)
    (nested / "data.yaml").write_text("path: .\n", encoding="utf-8")
    assert discover_yaml_in_tree(tmp_path / "imported") is not None

    found, notes = scan_yolo_folder(tmp_path / "missing")
    assert found is None
    assert notes