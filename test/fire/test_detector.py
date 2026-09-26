import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from app.fire.detector import FireDetector


class DetectorTests(unittest.TestCase):
    def test_missing_weights_has_actionable_error(self):
        with patch.dict(os.environ, {'FIRE_MODEL_PATH': '/not-present/roommind-fire.pt'}):
            detector = FireDetector()
        with self.assertRaisesRegex(RuntimeError, 'FIRE_MODEL_PATH'):
            detector.load()

    def test_regular_object_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.pt'
            path.touch()
            module = types.ModuleType('ultralytics')
            module.YOLO = lambda _: types.SimpleNamespace(names={0: 'person', 1: 'chair'})
            with patch.dict(os.environ, {'FIRE_MODEL_PATH': str(path)}), patch.dict(sys.modules, {'ultralytics': module}):
                detector = FireDetector()
                with self.assertRaisesRegex(RuntimeError, 'fire 和 smoke'):
                    detector.load()
                self.assertIsNone(detector._model)

    def test_mapping_uses_model_labels_not_assumed_ids(self):
        def scalar(value):
            return types.SimpleNamespace(item=lambda: value)
        box = types.SimpleNamespace(cls=scalar(1), conf=scalar(.95), xyxy=[types.SimpleNamespace(tolist=lambda: [1, 2, 3, 4])])
        model = types.SimpleNamespace(names={0: 'smoke', 1: 'fire'}, predict=lambda *a, **k: [types.SimpleNamespace(names={0: 'smoke', 1: 'fire'}, boxes=[box])])
        detector = FireDetector()
        detector._model = model
        self.assertEqual(detector.detect(object()), [{'label': 'fire', 'score': .95, 'bbox': [1, 2, 3, 4]}])
