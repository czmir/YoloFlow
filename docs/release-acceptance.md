# YoloFlow v0.1.0 release acceptance

**When:** 2026-11-24 14:43–14:44 (Asia/Shanghai, UTC+8)  
**Host:** Cursor cloud agent VM, Linux 6.12.113+ x86_64, **CPU-only** (no CUDA)  
**CPU:** 4× Intel Xeon, ~16 GiB RAM  
**Scripts:** `python scripts/acceptance_release.py` (exit 0), `pytest` 49 passed, Playwright `e2e/test_ui.py` 1 passed  
**Raw JSON:** `/agent/repos/YoloFlow/data/acceptance-release.json` (gitignored)

This VM is Linux. Windows start is documented in the repo as `scripts/setup.ps1` + `scripts/start-dev.ps1` (not executed here).

## Smoke vs formal eval

**COCO128 + 1 epoch is not generalization proof.** It only shows the import → train → val metrics → register → ONNX → infer loop runs on a public toy set. Do not cite these mAP numbers as a dataset-level result.

## Hardware and versions

| Item | Value |
| --- | --- |
| Python | 3.12.3 |
| torch | 2.5.1+cpu |
| torchvision | 0.20.1+cpu |
| ultralytics | 8.3.70 |
| onnxruntime | 1.20.1 |
| onnx | 1.17.0 |
| fastapi | 0.115.12 |
| numpy | 1.26.4 |
| CUDA | unavailable; device `cpu` |

Pinned in `backend/requirements.txt` so local == CI.

## Dataset

| Field | Value |
| --- | --- |
| Public set | Ultralytics COCO128 (`coco128.zip`) |
| Train images | 128 |
| Val images | 128 (same split in the stock yaml) |
| Train labels | 929 |
| Validation | `valid=true` |
| Warnings | missing labels for `000000000250.jpg`, `000000000508.jpg`; orphan labels `000000000656.txt`, `000000000659.txt` |
| Import + validate | **0.04 s** |

## Train / val / register / ONNX

Params: `yolov8n.pt`, epochs **1**, batch **4**, imgsz **320**, device **cpu**.

| Field | Value |
| --- | --- |
| Job | 57, `succeeded` |
| Wall time | **16.51 s** (Ultralytics elapsed **16.13 s**) |
| mAP50 | **0.505** |
| mAP50-95 | **0.379** |
| Precision | **0.624** |
| Recall | **0.468** |
| Weights | `data/workspaces/job_57/train/weights/best.pt` |
| ONNX | `data/workspaces/job_57/model.onnx` |
| Registry | registered on train success |

`scripts/acceptance_train.py` now **fails (exit 3)** if ONNX compare is skipped after a successful train.

## PyTorch vs ONNX Runtime (fixed image)

Image: `coco128/images/train2017/000000000009.jpg`. Both backends `imgsz=320`.

| Backend | Detections | Latency |
| --- | --- | --- |
| PyTorch | 3 | **50.3 ms** |
| ONNX (Ultralytics ORT) | 4 | **43.6 ms** |
| Mean max IoU (PT → ONNX) | **0.993** | |
| Count diff (PT − ONNX) | −1 | |

PT boxes (class, conf, xyxy):

- bowl 0.943 — `[0.0, 193.4, 622.6, 480.0]`
- bowl 0.828 — `[1.3, 2.4, 632.7, 435.4]`
- broccoli 0.327 — `[256.0, 241.2, 566.8, 477.3]`

ONNX extra box: bowl 0.440 (the other three match PT with high IoU).

## Normal vs tiled large image

Constructed `data/runs/large_tile.jpg` by 3×3 tiling the same JPEG → **1920×1440**.

| Mode | Boxes | Latency | Notes |
| --- | --- | --- | --- |
| Full frame, imgsz=320 | **0** | 48.7 ms | whole canvas letterboxed to 320; objects too small |
| Tiled, tile 320, overlap 0.2 | **104** | 739.5 ms | 48 tiles; 3×3 repeats plus overlap keep extras after class-NMS |

Tiling is doing work the full 320px pass does not. Counts are not a cleanliness claim (repeated pattern + overlap).

## Pause, checkpoint, resume, cancel, crash recovery

| Check | Result |
| --- | --- |
| Real pause + resume | Job 58, 2 epochs. Cooperative pause at epoch end after **5.75 s** (`paused`). Unpause + run → `succeeded` in **18.79 s**. `last.pt` stored as `resume_from`. |
| Cancel | Unit tests: pending cancel → `cancelled`; succeeded cancel → HTTP 409. Running cancel is cooperative (`JobCancelled` at epoch end), same hook as pause. Not a second multi-epoch wall-clock cancel in this run. |
| Crash / restart | Job 59 left `running` with dead lease / no heartbeat. `recover_orphan_jobs` re-queued it (`recovered: true`). |

In-process queue and `worker/main.py` share `claim_pending_job` compare-and-swap (`pending` → `running`). Compose disables the API embedded worker so only the worker container claims.

## Image / video / batch / export

| Mode | Result |
| --- | --- |
| Batch (3 COCO JPEGs) | 3, 1, 2 boxes; 38–49 ms each; **0.17 s** total |
| Video | 4 sampled frames from a 4-frame mp4, **88.2 ms** |
| Export | JSON `detections.json`, **599 bytes** |

## Standalone ORT service

`python deploy/ort_server/main.py` with `YOLOFLOW_ORT_MODEL=.../job_57/model.onnx`, port 8001.

| Call | Result |
| --- | --- |
| `GET /health` | `loaded: true`, `imgsz: 320` |
| `POST /v1/detect` on `000000000009.jpg` | HTTP **200**, **4** detections (bowl/bowl/bowl/broccoli), conf 0.948 / 0.756 / 0.440 / … |

## Playwright E2E

Stack: FastAPI `:8000` + Vite `:5173`. Chromium headless.

`pytest e2e/test_ui.py -m e2e` → **1 passed** (nav Inference / Datasets / Training / Experiments / Resources; WebSocket status text visible).

Not in GitHub Actions (no browser on the default CI job).

## Tests / CI

| Suite | Result |
| --- | --- |
| `pytest` (backend, this VM) | **49 passed** |
| Frontend `tsc` + `vite build` | OK |
| GitHub Actions before this release | failed on `test_resources_websocket_payload` (`concurrent.futures.CancelledError`) at [run 45337009530](https://github.com/czmir/YoloFlow/actions/runs/45337009530) |

Engineering fixes in this release: WS disconnect/cancel/heartbeat cleanup, pinned deps, atomic job claim, JobCancelled/JobPaused (no substring matching), zip path-traversal + weight-path limits, tiled NMS tests, ONNX skip is not success.

## Reproduce

```bash
bash scripts/setup.sh
source .venv/bin/activate
export PYTHONPATH="$(pwd)/backend"
export YOLO_AUTOINSTALL=false
python scripts/acceptance_release.py
bash scripts/verify-quickstart.sh
```
