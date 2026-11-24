# YoloFlow 技术讲义（给 leo）

面向仓库作者的源码地图：为什么这样设计、哪些地方容易问、以及已经踩过的真实坑。对应仓库 `main` 上的 v0.2 工程，不是论文。

面试时可以按这条线讲：**单机 YOLO 工作台 = 校验过的数据集 + 可抢占的训练队列 + 实验/模型注册 + 推理（含切片 NMS）+ 可选 ORT 侧车**。产品代码在 `backend/`、`worker/`、`frontend/`、`deploy/`，训练内核仍是 Ultralytics。

## 1. 仓库地图

| 路径 | 角色 | 先读的函数 |
| --- | --- | --- |
| `backend/app/main.py` | FastAPI 入口、lifespan 里开库、恢复孤儿任务、决定是否启动进程内 worker | `lifespan` |
| `backend/app/config.py` | `YOLOFLOW_` 前缀配置：SQLite URL、data 根、zip 上限、租约秒数、是否 embedded worker | `Settings` |
| `backend/app/database.py` / `models.py` | SQLAlchemy + SQLite WAL；`Dataset` / `TrainingJob` / `MetricPoint` / `ModelVersion` | 表定义 |
| `backend/app/api/datasets.py` | YAML / 扫目录 / zip 导入、校验、划分、标注预览 | `upload_zip`, `import_yaml`, `split_dataset` |
| `backend/app/api/training.py` | 建任务、队列快照、pause/cancel/unpause | `queue_status` |
| `backend/app/api/logs.py` | 日志 tail + SSE | `read_job_logs` |
| `backend/app/api/inference.py` | 单张 / 批量 / 视频 / 切片 / 导出 | `_resolve_weights`, `detect` |
| `backend/app/api/models.py` | 模型注册表、ONNX 再导出、PT vs ONNX 批量 IoU | `register` 路径在 trainer 里 |
| `backend/app/api/experiments.py` | 实验列表、对比曲线、混淆矩阵文件 | `_summary` |
| `backend/app/services/job_state.py` | 状态机 + CAS 抢占 + 租约恢复 | `claim_pending_job`, `recover_orphan_jobs` |
| `backend/app/services/ultralytics_trainer.py` | 真正调 `YOLO.train`、心跳、pause/cancel 回调、成功后导出 ONNX 并注册 | `run_training_job`, `TrainingQueue` |
| `backend/app/services/tiling.py` | 滑窗切图 + 类别感知 NMS | `generate_tiles`, `class_aware_nms` |
| `backend/app/services/infer_compare.py` | 全图 vs 切片的 IoU / 重复框 / GT 匹配（验收用，也可单测） | `match_detections_to_gt` |
| `backend/app/services/inference_engine.py` | PT / ONNX / tiled / video | `infer_tiled_detailed` |
| `backend/app/services/security.py` | zip 防穿越、权重路径白名单 | `safe_extract_zip`, `resolve_weight_path` |
| `backend/app/ws/resources.py` | 资源 WebSocket：发送/接收拆开，ping/pong | `resources_ws` |
| `worker/main.py` | 独立轮询进程，**同一套** `claim_pending_job` | `poll_loop` |
| `frontend/src/pages/*.tsx` | Datasets / Training / Experiments / Inference / Resources | `DatasetsPage`, `TrainingPage` |
| `deploy/ort_server/main.py` | 只加载 `.onnx` 的侧车，端口 8001 | `/v1/detect` |
| `tests/` | 默认 CI 跑这里；`e2e/` 是 Playwright，不进 GitHub Actions | `pytest.ini` 的 `e2e` marker |
| `scripts/` | `setup.sh`、`acceptance_v02.py`、`acceptance_e2e.sh`、`acceptance_compose.sh` | — |

数据都落在仓库 `data/`（gitignored）：SQLite、`datasets/`、`workspaces/job_<id>/`、上传 zip。

## 2. 为什么是这个形状

目标是**一个人能在 CPU 上跑通闭环**，不是分布式训练器。

- **SQLite 而不是 Postgres**：单机、零运维。并发模型因此必须自己做：两个进程可以跑两个不同 job，但不能跑同一行。
- **Ultralytics 做训练/导出**：自己写 YOLO 反传没有意义。YoloFlow 的价值在队列、校验、注册、UI、切片 NMS、ORT 侧车。
- **进程内队列 + 可选独立 worker**：开发时 `YOLOFLOW_EMBEDDED_WORKER=true`（API 自己 `TrainingQueue.kick()`）。Compose 里 API 设 `false`，只让 `worker` 容器抢任务，避免双跑。
- **合作式 pause/cancel**：Ultralytics 的 batch 循环不好强杀。`JobControlCallback.on_train_epoch_end` 在 epoch 结束看 `cancel_requested` / `pause_requested`。面试时要主动说：**不是 mid-batch 抢占**。
- **MIT 自己的代码，AGPL 是 Ultralytics**：`NOTICE.md` + README 许可证表。分发带 Ultralytics 的服务可能触发 AGPL 源码提供义务。不要把 MIT 徽章说成「整个栈都是 MIT」。

## 3. 任务队列 vs claim（最容易讲错）

两条路径最终都要调用 `run_training_job(job_id)`：

1. **API 进程内**：`TrainingQueue.kick()` → `claim_pending_job` → 后台线程 `_run`。
2. **独立 worker**：`worker/main.py` 的 `poll_loop` → 同样 `claim_pending_job` → `run_training_job`。

关键不在「有个队列数组」，而在 **CAS**：

```
claim_pending_job
  选出 status=pending 且最早 created_at 的行
  _preflight_resume：resume_from 文件必须存在，否则直接 failed
  _cas_pending_to_running：UPDATE ... WHERE id=? AND status='pending'
  rowcount == 1 才算抢到；否则循环抢下一行
```

`tests/test_claim_and_recovery.py::test_claim_is_exclusive` 用两个线程同时 claim，断言只有一个 winner。v0.1.0 之前若只在 Python 里 `job.status = running` 再 commit，两个 worker 会双跑。修复就是这条 SQL `WHERE status='pending'`。

`TrainingQueue.enqueue` 现在忽略传入的 job id，统一走 CAS。这是有意的：API 建任务只负责 insert `pending`，谁跑由 claim 决定。

Compose：`docker-compose.yml` 里 api 的 `YOLOFLOW_EMBEDDED_WORKER=false`，worker 服务 `command: python worker/main.py`。两边共享 `./data` 卷和同一 SQLite 文件。

## 4. Pause / resume / 崩溃恢复

状态集合见 `JobStatus`。合法迁移在 `job_state.py`：

| 动作 | 函数 | 行为 |
| --- | --- | --- |
| 取消 pending | `apply_cancel` | 立刻 `cancelled` |
| 取消 running | `apply_cancel` | 只设 `cancel_requested=True`，等 epoch 结束 `JobCancelled` |
| 暂停 pending | `apply_pause` | 立刻 `paused` |
| 暂停 running | `apply_pause` | `pause_requested=True` → 回调里写 `paused` 并抛 `JobPaused`，同时记下 `last.pt` 到 `resume_from` |
| 继续 | `apply_unpause` | `paused` → `pending`，清 flag，再被 claim |
| 崩溃 | `recover_orphan_jobs` | `running` 且心跳早于 `YOLOFLOW_JOB_LEASE_SECONDS`（默认 90s）→ 改回 `pending`；若有 `train/weights/last.pt` 则设 `resume_from` |

心跳：`run_training_job` 里 `_heartbeat_loop` 每 15s `touch_heartbeat`（还要求 `claimed_by` 是自己）。lifespan 和 worker 启动都会 `recover_orphan_jobs`。

**不要用异常消息字符串判断取消/暂停。** 以前踩过 substring 匹配。现在是 `JobCancelled` / `JobPaused` 类型，`is_control_exception` 沿 `__cause__` / `__context__` 往上找，因为 Ultralytics 可能把你的异常包一层。

Resume 时 `model.train(..., resume=True)`，权重路径还要过 `resolve_weight_path`：只允许 Ultralytics 内置名（`yolov8n.pt`）或 `data/` 下的文件。

## 5. 切片检测与类别感知 NMS

大图直接 `imgsz=320` 会 letterbox 整张图，小目标容易丢。`generate_tiles(w,h,tile_size,overlap)` 按步长 `tile_size*(1-overlap)` 滑窗，末块贴边，最后 `dict.fromkeys` 去重。

每个 tile 上 `YOLO.predict`，框用 `remap_tile_box` 加原点。重叠区域会重复检出，所以要 **全局 NMS**。

`class_aware_nms`：**同类** IoU 超过阈值才抑制，**不同类**即使重叠也保留（人抱着杯子不该互杀）。`nms_boxes` 是按分数贪心。验收侧 `infer_compare.duplicate_report` 统计 NMS 前/后的同类 IoU>0.5 对数，用来解释「切片为什么框更多、NMS 砍掉了多少」。

`infer_tiled` 保持原 API；`infer_tiled_detailed` 多返回 pre/post-NMS 统计，给 `scripts/acceptance_v02.py` 用。单测锁行为：`tests/test_tiling.py`、`tests/test_infer_compare.py`（不加载权重，CI 可跑）。

## 6. ONNX 路径

训练成功：`export_to_onnx(best, workspace/model.onnx)` → `register_job_model` 写 `model_versions` + `onnx_export_records`（SHA-256）。

推理：`_resolve_weights` 优先 `model_version_id`，否则 job 的 `best_weights`，否则请求里的路径（仍受 `resolve_weight_path` 限制）。`compare_pytorch_onnx` 用 ONNX 输入空间尺寸对齐 PT 的 `imgsz`，再算 PT 框到 ONNX 框的 mean max IoU。

v0.1.0 修过一个验收谎言：训练成功但没 ONNX 对比时，`acceptance_train.py` 仍返回 0。现在缺权重/ONNX 会 **exit 3**，不再把 skip 当成功。

独立服务：`deploy/ort_server/main.py`，环境变量 `YOLOFLOW_ORT_MODEL`，Compose profile `ort`。它**不**走训练队列，只做 HTTP detect。

## 7. 安全（上传与权重）

`safe_extract_zip` / `zip_member_path`：拒绝绝对路径、`..`、文件数和未压缩体积上限（`YOLOFLOW_MAX_ZIP_FILES` / `MAX_ZIP_UNCOMPRESSED_MB`）。`tests/test_security.py` 覆盖 traversal HTTP 400/413。

`resolve_weight_path`：防止 `../../etc/passwd` 式权重。内置名用正则 `_BUILTIN_WEIGHTS`。

这不是完整多租户安全模型：没有账号系统，CORS 默认本机 Vite 端口。面试时别吹成零信任。

## 8. 前端

Vite + React Router。`frontend/vite.config.ts` 把 `/api` 和 `/ws` 代理到 `YOLOFLOW_API_PROXY`（默认 `http://127.0.0.1:8000`）。Compose 里前端容器必须代理到 `http://api:8000`，否则会打到容器自己的 localhost。

页面：

- `Datasets.tsx`：YAML / 扫目录 / zip；`data-testid` 给 E2E 用。
- `Training.tsx`：3s 轮询 jobs + queue；Logs 打 `/training/jobs/{id}/logs`。
- `Experiments.tsx`：Recharts 对比 `val_map50` / loss。
- `Inference.tsx`：canvas 画框；tiled checkbox。
- `Resources.tsx`：WebSocket 重连退避，ping → pong。

E2E 在 `e2e/test_flow.py`：真的上传 zip、建 1 epoch 任务、看 Experiments、跑全图和切片推理。默认 CI **不**装浏览器，所以 GitHub Actions 只跑 `tests/` + 前端 `tsc`/`vite build`。

## 9. 测试与 CI

| 层 | 命令 | 是否在 GitHub Actions |
| --- | --- | --- |
| 后端单测 | `cd backend && PYTHONPATH=../backend pytest`（`backend/pytest.ini` 的 `testpaths = ../tests`） | 是，CPU torch 2.5.1 |
| 前端 | `npm run typecheck && npm run build` | 是 |
| Playwright | `bash scripts/acceptance_e2e.sh` | 否 |
| 独立划分评估 | `python scripts/acceptance_v02.py` | 否（要下载 VOC、要训练） |
| Compose | `bash scripts/acceptance_compose.sh` | 否 |
| Windows `setup.ps1` | 本 Linux CI/云主机 **NOT RUN** | — |

`.github/workflows/ci.yml` 钉死 `torch==2.5.1` CPU 轮 + `backend/requirements.txt`。本地请走 `scripts/setup.sh`，不要随手 `pip install ultralytics` 漂版本。

## 10. 真实修过的 bug（面试可以讲因果）

1. **资源 WebSocket `CancelledError`**（v0.1.0 CI 红过，run 45337009530）  
   断开时 send/receive 和心跳缠在一起，Starlette 取消任务冒到 TestClient。拆成 `_sender` / `_receiver`，`CancelledError` 里 `stop.set()`，finally 里 cancel + `gather(..., return_exceptions=True)`。见 `backend/app/ws/resources.py`。

2. **双 worker 抢同一 job**  
   非原子 status 更新。改成 SQL CAS，`test_claim_is_exclusive` 锁住。

3. **pause/cancel 用字符串匹配**  
   Ultralytics 包一层异常就失效。改成异常类型 + `is_control_exception`。

4. **zip 路径穿越**  
   上传解压未规范化。`zip_member_path` + `is_within`。

5. **ONNX skip 当成功**  
   验收脚本看到训练成功就 exit 0。现在缺对比返回 3。

6. **COCO128 1 epoch 的 mAP 当论文数字**  
   官方 coco128 的 train/val 是同一 128 张。v0.2 改用 **PASCAL VOC 2007 官方 `ImageSets/Main/train.txt` vs `val.txt`** 的子集做独立评估（`scripts/prepare_public_split.py` + `acceptance_v02.py`）。1 epoch coco128 只保留为冒烟。

7. **Compose 前端代理打到容器 localhost**  
   Vite proxy 写死 `127.0.0.1:8000`。v0.2 改为 `YOLOFLOW_API_PROXY`，compose 里设 `http://api:8000`。

8. **zip 导入能校验、训练却找不到图**  
   归档里的 `data.yaml` 常用 `path: .`。`import_dataset_from_yaml` 按 yaml 所在目录解析，所以 UI 显示 valid；`YOLO.train` 却相对进程 CWD（经常是 `backend/`）找图。v0.2 在 `materialize_yaml_root` 里把相对 `path` 写成 yaml 目录的绝对路径。

## 11. 部署怎么记

```text
浏览器 :5173  --proxy /api,/ws-->  FastAPI :8000
                                   ├ SQLite data/yoloflow.db
                                   ├ 可选 embedded TrainingQueue
                                   └ 推理 / 数据集 API
worker 容器  --同一 DB-->  claim + YOLO.train
ort 容器 :8001  --只读 onnx-->  /v1/detect
```

Linux：`bash scripts/setup.sh` 然后 `scripts/start-dev.sh`。  
Windows：`scripts/setup.ps1` / `start-dev.ps1`（本仓库的 Linux CI **没有**跑过）。  
Docker：`docker compose up --build`。

## 12. 一句话答辩

> YoloFlow 把 Ultralytics 包成可运营的单机工作台：数据集校验、CAS 任务锁、epoch 边界的暂停续训、实验指标、ONNX 注册，以及带类别感知 NMS 的切片推理。难处不在 backbone，而在「两个进程不能双跑同一行、崩溃能按 last.pt 续、验收数字不能拿重叠 val 充数」。
