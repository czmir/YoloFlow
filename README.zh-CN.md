# YoloFlow

[English](README.md)

面向个人与小团队的轻量化 YOLO 平台：数据集校验与标注预览、CPU 训练与任务日志、模型注册与 ONNX 对比、实验图表、在线推理（批量/视频/大图切片）。

## 快速开始

```bash
bash scripts/setup.sh
source .venv/bin/activate   # 若已创建 venv
export PYTHONPATH="$(pwd)/backend"
uvicorn app.main:app --app-dir backend --reload --port 8000
cd frontend && npm run dev
bash scripts/verify-quickstart.sh
python scripts/acceptance_train.py
```

## 架构

参见英文 README 中的 Mermaid 图（FastAPI + SQLite + React + 可选 Worker + ORT 侧车）。

## 资源监控

WebSocket：`/ws/resources`，约 1s 推送 CPU/内存与队列；客户端可发送 `ping` 收到 `pong`。

## 训练日志

`GET /api/v1/training/jobs/{id}/logs?offset=&limit=`，训练页 **Logs** 按钮查看。

## 许可证

MIT