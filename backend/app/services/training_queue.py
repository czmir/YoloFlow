from __future__ import annotations

from app.services.ultralytics_trainer import TrainingQueue, training_queue


def get_queue() -> TrainingQueue:
    return training_queue
