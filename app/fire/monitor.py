"""One camera worker, bounded event memory and non-blocking semantic review."""
import copy
import json
import logging
import os
import threading
import time
from collections import deque
from urllib.request import Request, urlopen

from app.fire.detector import FireDetector
from app.fire.rules import FireRules, iso

log = logging.getLogger(__name__)


class FireMonitor:
    def __init__(self, camera_lock, detector=None):
        self.camera_lock = camera_lock
        self.detector = detector or FireDetector()
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.review_lock = threading.Lock()
        self.events = deque(maxlen=100)
        self.rules = None
        self.state = 'stopped'
        self.error = None
        self.last_frame_at = None
        self.detections = []
        self.source = None
        self.location = None

    def start(self, source, location, semantics=True):
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise RuntimeError('火警监测已在运行或正在停止')
            if not self.camera_lock.acquire(blocking=False):
                raise RuntimeError('摄像头正在使用，请先关闭预览')
            self.stop_event.clear()
            self.rules = FireRules(location)
            self.source, self.location = source, location
            self.state, self.error, self.last_frame_at = 'starting', None, None
            self.detections = []
            self.thread = threading.Thread(target=self._run, args=(source, semantics), daemon=True)
            try:
                self.thread.start()
            except Exception:
                self.camera_lock.release()
                self.state = 'error'
                raise
        return self.status()

    def stop(self):
        with self.lock:
            self.stop_event.set()
            if self.thread and self.thread.is_alive():
                self.state = 'stopping'
        return self.status()

    def status(self):
        with self.lock:
            now = time.time()
            return copy.deepcopy({'state': self.state, 'error': self.error,
                'source': self.source, 'location': self.location,
                'last_frame_at': iso(self.last_frame_at) if self.last_frame_at else None,
                'stale': self.last_frame_at is None or now - self.last_frame_at > 10,
                'detections': self.detections,
                'risks': self.rules.snapshot(now) if self.rules else [],
                'events': list(self.events)})

    def _run(self, source, semantics):
        cap = None
        try:
            from app.camera.capture import open_capture, parse_source, iter_frames
            self.detector.load()
            if self.stop_event.is_set():
                return
            cap = open_capture(parse_source(source), width=1280, height=720)
            with self.lock:
                self.state = 'running'
            for frame in iter_frames(cap, max_fps=1):
                if self.stop_event.is_set():
                    break
                now = time.time()
                detections = self.detector.detect(frame)
                with self.lock:
                    self.last_frame_at = now
                    self.detections = detections
                    events = self.rules.update(detections, now)
                    for event in events:
                        event['review_status'] = 'disabled' if not semantics else 'queued'
                        self.events.append(event)
                # Publish the rule alert before any potentially slow VLM / Agent request.
                if events and semantics:
                    if self.review_lock.acquire(blocking=False):
                        threading.Thread(target=self._review, args=(frame.copy(), events), daemon=True).start()
                    else:
                        with self.lock:
                            for event in events:
                                event['review_status'] = 'skipped_busy'
            if not self.stop_event.is_set():
                raise RuntimeError('摄像头画面中断，火警监测已停止')
        except Exception as exc:
            log.exception('Fire monitor failed')
            with self.lock:
                self.state, self.error = 'error', str(exc)
        finally:
            if cap is not None:
                cap.release()
            self.camera_lock.release()
            with self.lock:
                if self.state != 'error':
                    self.state = 'stopped'

    def _review(self, frame, events):
        try:
            import cv2
            ok, encoded = cv2.imencode('.jpg', frame)
            if not ok:
                raise RuntimeError('无法编码火警画面')
            import base64
            ollama = os.environ.get('FIRE_OLLAMA_URL', 'http://127.0.0.1:11434').rstrip('/')
            payload = {'model': 'qwen3-vl:2b-instruct', 'stream': False,
                'messages': [{'role': 'user', 'content':
                    '描述可见的疑似火焰、烟雾、灶台及附近是否有人。区分可见事实与不确定推测，不要断言火灾已确认。',
                    'images': [base64.b64encode(encoded.tobytes()).decode()]}]}
            semantic = self._post(ollama + '/api/chat', payload)['message']['content']
            with self.lock:
                history = [{k: copy.deepcopy(v) for k, v in event.items()
                            if k not in ('agent_review', 'semantic', 'review_error')}
                           for event in list(self.events)[-10:]]
                for event in events:
                    event['semantic'] = semantic
                    event['review_status'] = 'agent_pending'
            agent = os.environ.get('FIRE_AGENT_URL', 'http://127.0.0.1:8001').rstrip('/')
            review = self._post(agent + '/fire/review', {
                'events': copy.deepcopy(events), 'semantic': semantic, 'history': history})
            with self.lock:
                for event in events:
                    event['agent_review'] = review
                    event['review_status'] = 'complete'
        except Exception as exc:
            with self.lock:
                for event in events:
                    event['review_status'] = 'failed'
                    event['review_error'] = str(exc)
        finally:
            self.review_lock.release()

    @staticmethod
    def _post(url, payload):
        request = Request(url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=45) as response:
            return json.load(response)
