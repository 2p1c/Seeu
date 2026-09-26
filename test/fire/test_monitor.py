import sys
import threading
import types
import unittest
from unittest.mock import patch

from app.fire.monitor import FireMonitor


class Detector:
    def load(self):
        pass

    def detect(self, frame):
        return [{'label': 'fire', 'score': .95, 'bbox': [0, 0, 1, 1]}]


class MonitorTests(unittest.TestCase):
    def test_camera_busy_does_not_start(self):
        lock = threading.Lock()
        lock.acquire()
        monitor = FireMonitor(lock, Detector())
        with self.assertRaisesRegex(RuntimeError, '摄像头正在使用'):
            monitor.start('0', '厨房', False)
        self.assertEqual(monitor.status()['state'], 'stopped')

    def test_model_failure_releases_camera(self):
        class Broken(Detector):
            def load(self):
                raise RuntimeError('missing weights')
        lock = threading.Lock()
        monitor = FireMonitor(lock, Broken())
        # Camera module import is substituted to avoid hardware dependencies.
        module = types.ModuleType('app.camera.capture')
        module.open_capture = module.parse_source = module.iter_frames = lambda *a, **k: None
        with patch.dict(sys.modules, {'app.camera.capture': module}):
            monitor.start('0', '厨房', False)
            monitor.thread.join(2)
        self.assertEqual(monitor.status()['state'], 'error')
        self.assertIn('missing weights', monitor.status()['error'])
        self.assertFalse(lock.locked())

    def test_camera_disconnect_keeps_alert_and_reports_error(self):
        module = types.ModuleType('app.camera.capture')
        cap = types.SimpleNamespace(release=lambda: None)
        module.open_capture = lambda *a, **k: cap
        module.parse_source = lambda source: source
        module.iter_frames = lambda *a, **k: iter([object()])
        lock = threading.Lock()
        monitor = FireMonitor(lock, Detector())
        with patch.dict(sys.modules, {'app.camera.capture': module}):
            monitor.start('0', '厨房', False)
            monitor.thread.join(2)
        result = monitor.status()
        self.assertEqual(result['state'], 'error')
        self.assertEqual(len(result['events']), 1)
        self.assertEqual(result['events'][0]['review_status'], 'disabled')
        self.assertFalse(lock.locked())
        result['events'].clear()
        self.assertEqual(len(monitor.status()['events']), 1)

    def test_semantic_failure_keeps_alert(self):
        module = types.ModuleType('cv2')
        module.imencode = lambda *a: (True, types.SimpleNamespace(tobytes=lambda: b'image'))
        monitor = FireMonitor(threading.Lock(), Detector())
        event = {'id': 'one', 'decision': 'alert', 'review_status': 'queued'}
        monitor.events.append(event)
        monitor.review_lock.acquire()
        with patch.dict(sys.modules, {'cv2': module}), patch.object(monitor, '_post', side_effect=TimeoutError('offline')):
            monitor._review(object(), [event])
        self.assertEqual(event['decision'], 'alert')
        self.assertEqual(event['review_status'], 'failed')
        self.assertFalse(monitor.review_lock.locked())

    def test_slow_review_does_not_block_camera_loop(self):
        module = types.ModuleType('app.camera.capture')
        module.open_capture = lambda *a, **k: types.SimpleNamespace(release=lambda: None)
        module.parse_source = lambda source: source
        frame = types.SimpleNamespace(copy=lambda: object())
        module.iter_frames = lambda *a, **k: iter([frame, frame])
        monitor = FireMonitor(threading.Lock(), Detector())
        release_review = threading.Event()
        review_started = threading.Event()
        review_done = threading.Event()

        def slow_review(*args):
            review_started.set()
            release_review.wait(2)
            monitor.review_lock.release()
            review_done.set()

        try:
            with patch.dict(sys.modules, {'app.camera.capture': module}), patch.object(monitor, '_review', side_effect=slow_review):
                monitor.start('0', '厨房', True)
                self.assertTrue(review_started.wait(1))
                monitor.thread.join(1)
                self.assertFalse(monitor.thread.is_alive())
                self.assertEqual(len(monitor.status()['events']), 1)
        finally:
            release_review.set()
            review_done.wait(2)


if __name__ == '__main__':
    unittest.main()
