from pathlib import Path

from app.services.dataset_validator import validate_yolo_dataset


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
    yaml.write_text("path: {}\ntrain: images/train\nval: images/train\nnames: [obj]\n".format(root))

    stats, validation = validate_yolo_dataset(yaml)
    assert validation.valid
    assert stats.num_images_train == 1
    assert stats.num_labels_train == 1