"""HTTP contract tests; requires normal perception dependencies, no model weights."""
import unittest
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
    from app import main
except ImportError:
    TestClient = None


@unittest.skipIf(TestClient is None, 'Install perception dependencies for HTTP tests')
class FireAPITests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def test_initial_status_and_stop(self):
        response = self.client.get('/api/fire/status')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['state'], 'stopped')
        self.assertTrue(response.json()['stale'])
        self.assertEqual(self.client.post('/api/fire/stop').status_code, 200)

    def test_invalid_source_or_location_rejected(self):
        for body in ({'source': 'http://example.com/video'}, {'location': '  '}, {'source': ''}):
            self.assertEqual(self.client.post('/api/fire/start', json=body).status_code, 422)

    def test_busy_camera_returns_conflict_without_starting(self):
        main._camera_lock.acquire()
        try:
            response = self.client.post('/api/fire/start', json={'location': '厨房'})
            self.assertEqual(response.status_code, 409)
            with patch.object(main, 'list_video_devices', side_effect=AssertionError('must not probe busy camera')):
                self.assertEqual(self.client.get('/api/camera/devices').status_code, 200)
        finally:
            main._camera_lock.release()

    def test_corrupt_image_rejected(self):
        for data in (b'', b'not an image'):
            response = self.client.post('/api/fire/detect', files={'image': ('test.jpg', data, 'image/jpeg')})
            self.assertEqual(response.status_code, 400)

    def test_image_detection_does_not_change_monitor_history(self):
        import cv2
        import numpy as np
        _, image = cv2.imencode('.jpg', np.zeros((8, 8, 3), dtype=np.uint8))
        before = main.fire_monitor.status()['events']
        with patch.object(main.fire_monitor.detector, 'detect', return_value=[{'label': 'fire', 'score': .9, 'bbox': [0, 0, 4, 4]}]):
            response = self.client.post('/api/fire/detect', files={'image': ('test.jpg', image.tobytes(), 'image/jpeg')})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['mode'], 'single_image')
        self.assertEqual(main.fire_monitor.status()['events'], before)
