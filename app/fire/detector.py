"""Adapter for e1250/safety_detection (Ultralytics YOLO)."""
import os
import threading
from pathlib import Path


class FireDetector:
    def __init__(self):
        self.path = Path(os.environ.get('FIRE_MODEL_PATH',
            str(Path(__file__).resolve().parents[2] / 'models/fire/model.pt')))
        self._model = None
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self._model is not None:
                return
            if not self.path.is_file():
                raise RuntimeError(f'火警模型不存在：{self.path}。请下载 e1250/safety_detection 权重，并设置 FIRE_MODEL_PATH。')
            from ultralytics import YOLO
            model = YOLO(str(self.path))
            labels = {str(v).lower() for v in model.names.values()}
            if not {'fire', 'smoke'}.issubset(labels):
                raise RuntimeError('火警权重必须包含 fire 和 smoke 类别，不能使用普通 YOLO 权重。')
            self._model = model

    def detect(self, frame):
        self.load()
        with self._lock:
            result = self._model.predict(frame, conf=0.4, imgsz=640, verbose=False)[0]
            detections = []
            for box in result.boxes:
                label = str(result.names[int(box.cls.item())]).lower()
                if label in ('fire', 'smoke'):
                    detections.append({'label': label, 'score': float(box.conf.item()),
                                       'bbox': box.xyxy[0].tolist()})
            return detections
