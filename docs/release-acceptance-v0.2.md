# YoloFlow v0.2.0 release acceptance

**When:** 2026-11-24 21:15–21:24 (Asia/Shanghai, UTC+8)  
**Host:** Cursor cloud agent VM, Linux 6.12.113+ x86_64, **CPU-only** (no CUDA)  
**CPU:** 4× Intel Xeon, ~16 GiB RAM  
**HEAD this package was measured on:** working tree of `cursor/v020-release-eval-efb1` (to be tagged `v0.2.0` after merge)

This VM is Linux. Windows `scripts/setup.ps1` + `scripts/start-dev.ps1` were **NOT RUN** (no self-hosted Windows worker is connected).

Raw sanitized files: [docs/acceptance-artifacts/](acceptance-artifacts/).

## Checks

| Check | Result | Evidence |
| --- | --- | --- |
| Independent-split train/val eval | **PASS** | COCO 2017 train2017 vs val2017 subset (40 / 20, overlap 0). VOC 2007 official ImageSets (48 / 24, overlap 0). |
| Meaningful val mAP (not COCO128-overlap 1-epoch) | **PASS** | COCO mini 10 epochs: mAP50 **0.587**, mAP50-95 **0.438**. VOC mini (new 20-class head): mAP50 **0.220**. |
| Full vs tiled infer + NMS duplicates | **PASS** | 20 COCO val images + 2×2 mosaic. Tables below / `infer-compare.md`. Locked by `tests/test_infer_compare.py`. |
| Playwright E2E (upload → train → inspect → infer) | **PASS** | `e2e/test_flow.py` 2 passed. Train **succeeded**. Full infer 3 boxes / 47.1 ms; tiled 3 boxes / 45.2 ms (`pytorch-tiled`). |
| Docker Compose on this Linux VM | **PASS** | `docker compose up --build`: API `GET /health` `{"status":"ok"}`, datasets/jobs/models 200, frontend HTTP 200. |
| Windows start scripts | **NOT RUN** | This VM is Linux; no user Windows worker. |
| `docker compose --profile ort` | **NOT RUN** | Default compose was the required stack; ORT sidecar needs `YOLOFLOW_ORT_MODEL`. v0.1.0 already exercised `deploy/ort_server` locally. |
| Main CI | **(after merge)** | `.github/workflows/ci.yml` — local `pytest` **58 passed**, frontend `tsc` OK. |
| Zip `path: .` train bug | **PASS** | `materialize_yaml_root`; E2E train failed until this fix, then succeeded. |

## Hardware and versions

| Item | Value |
| --- | --- |
| Python | 3.12.3 |
| torch | 2.5.1+cpu |
| ultralytics | 8.3.70 |
| onnxruntime | 1.20.1 |
| fastapi | 0.115.12 |
| numpy | 1.26.4 |
| CUDA | unavailable; device `cpu` |

## Dataset A — COCO 2017 official splits (primary)

| Field | Value |
| --- | --- |
| Public set | COCO 2017 (`instances_train2017.json` / `instances_val2017.json` + JPEGs from `images.cocodataset.org`) |
| Split | official **train2017** vs **val2017** (image ids disjoint) |
| Subset | 40 train / 20 val, seed 42, classes with ≥1 box |
| Command | `python scripts/prepare_coco_subset.py && python scripts/acceptance_v02.py` |
| Train | yolov8n.pt, 10 epochs, batch 4, imgsz 320, CPU |
| Wall time | **25.27 s** (Ultralytics elapsed **24.88 s**) |
| mAP50 | **0.58688** |
| mAP50-95 | **0.43815** |
| Precision | **0.8207** |
| Recall | **0.4278** |
| Job | 15 `succeeded` |

These mAP numbers are Ultralytics val on the **held-out val2017 subset**, not COCO128’s overlapping 128.

## Dataset B — VOC 2007 official ImageSets (secondary)

| Field | Value |
| --- | --- |
| Public set | PASCAL VOC 2007 trainval (`VOCtrainval_06-Nov-2007.zip` via Ultralytics GitHub assets) |
| Split | `ImageSets/Main/train.txt` (2501) vs `val.txt` (2510), overlap **0**; subset 48 / 24 |
| Train | 10 epochs, batch 4, imgsz 320; **nc=20** replaces the COCO head |
| Wall time | **26.74 s** |
| mAP50 | **0.21994** |
| mAP50-95 | **0.17981** |
| Note | detect@0.25 is often empty; Ultralytics AP uses conf≈0.001. Honest under-trained 20-class head. |

## Full image vs tiled

Conf 0.25, tile 320, overlap 0.2, class-aware NMS IoU 0.5. Accuracy = greedy class-aware match to YOLO labels at IoU 0.5.

| Mode | mean latency (ms) | mean P@0.5 | mean R@0.5 | duplicate pairs (same-class IoU>0.5) |
| --- | ---: | ---: | ---: | --- |
| Full image | 44.01 | 0.857 | 0.558 | n/a (single pass) |
| Tiled + NMS | 114.68 | 0.454 | 0.575 | **9.8 pre → 0.0 post**; mean 6.75 boxes suppressed |

Mosaic of **four distinct** val images (1140×1152, 10 GT):

| Mode | boxes | ms | P@0.5 | R@0.5 | tiles | pre/post NMS boxes | dups pre/post |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Full imgsz=320 | 6 | 50.92 | 0.667 | 0.400 | 1 | — | — |
| Tiled | 18 | 399.87 | 0.222 | 0.400 | 25 | 22 / 18 | 4 / 0 |

Tiling raises recall a little on native val images and finds more boxes on the large mosaic; precision drops until NMS. NMS zeros same-class overlap pairs in this run.

Per-image rows: [acceptance-artifacts/infer-compare.md](acceptance-artifacts/infer-compare.md).

## Playwright E2E

Stack: FastAPI `:8000` + Vite `:5173`, Chromium headless. `bash scripts/acceptance_e2e.sh`.

| Step | Result |
| --- | --- |
| Nav (Inference/Datasets/Training/Experiments/Resources) | PASS (`e2e/test_ui.py`) |
| Upload coco8 zip via UI | PASS (`uploaded zip`; 4 train / 4 val) |
| Start 1-epoch job | PASS |
| Wait until succeeded | PASS (~8.8 s elapsed in UI) |
| Experiments + Resources WS | PASS |
| Detect image (trained job) | PASS `backend=pytorch latency=47.1ms detections=3` |
| Tiled detect | PASS `backend=pytorch-tiled latency=45.2ms detections=3 tiles=1` |

First E2E train attempt **failed** (`path: .` resolved against `backend/`). Fixed by `materialize_yaml_root`; re-run passed. Not skipped.

## Docker Compose

`bash scripts/acceptance_compose.sh` on this Linux VM (docker 29.1.3, compose 2.40.3).

| Call | Result |
| --- | --- |
| `GET /health` | `{"status":"ok","app":"YoloFlow"}` |
| `GET /api/v1/datasets` | 200 |
| `GET /api/v1/training/jobs` | 200 |
| `GET /api/v1/models` | 200 |
| Frontend `:5173` | HTTP 200 |
| Services | api, worker, frontend **Up** |

Logs (sanitized): `docs/acceptance-artifacts/compose-compose-logs.txt`.

## Tests / CI

| Suite | Result |
| --- | --- |
| `pytest` backend (`tests/`) | **58 passed** |
| Frontend `tsc` + `vite build` | OK (typecheck run; build in `verify-quickstart.sh`) |
| Playwright | 2 passed, not in GitHub Actions |
| Windows | NOT RUN |

## Reproduce

```bash
bash scripts/setup.sh
source .venv/bin/activate   # or user-site Python if venv unavailable
export PYTHONPATH="$(pwd)/backend"
export YOLO_AUTOINSTALL=false
python scripts/acceptance_v02.py
bash scripts/acceptance_e2e.sh
bash scripts/acceptance_compose.sh
```
