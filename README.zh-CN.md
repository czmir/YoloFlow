# YoloFlow

[English](README.md)

面向个人与小团队的轻量化 YOLO 视觉平台：YOLO 数据集导入与校验、CPU 友好训练、ONNX 导出、在线推理（含大图切片 + NMS）及简易实验图表。

参考 [LiteTrain](https://github.com/gaoxuyang914/litetrain) 的产品方向，**独立实现**（React 前端、内置训练队列，不复制其业务/UI 代码）。

## 快速开始

```bash
bash scripts/setup.sh
source .venv/bin/activate
export PYTHONPATH="$(pwd)/backend"
uvicorn app.main:app --app-dir backend --reload --port 8000
# 另开终端
cd frontend && npm run dev
python scripts/acceptance_train.py
```

## 推送到 GitHub

空仓库创建后：

```bash
git remote add yoloflow https://github.com/czmir/YoloFlow.git
git push -u yoloflow main
```

## 许可证

MIT