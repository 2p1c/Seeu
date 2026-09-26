"""Deterministic fire rules; no model/network dependency."""
from datetime import datetime, timezone
from uuid import uuid4


def iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


class FireRules:
    def __init__(self, location, *, candidate=0.4, immediate=0.8,
                 persistence=5.0, clear_after=10.0, max_gap=10.0, cooldown=60.0):
        self.location = location
        self.candidate = candidate
        self.immediate = immediate
        self.persistence = persistence
        self.clear_after = clear_after
        self.max_gap = max_gap
        self.cooldown = cooldown
        self.states = {}
        self.last_sample = None

    def update(self, detections, now):
        # An outage is not evidence of continuous fire or of recovery.
        if self.last_sample is not None and now - self.last_sample > self.max_gap:
            self.states.clear()
        self.last_sample = now
        notices = []
        for kind in ('fire', 'smoke'):
            evidence = [d for d in detections if d['label'] == kind and d['score'] >= self.candidate]
            state = self.states.get(kind)
            if not evidence:
                if state:
                    state['clear_since'] = state.get('clear_since', now)
                    if now - state['clear_since'] >= self.clear_after:
                        del self.states[kind]
                    elif not state['alerted']:
                        # Weak candidates require consecutive positive observations.
                        del self.states[kind]
                continue
            if state is None:
                state = {'id': str(uuid4()), 'start': now, 'last': now,
                         'alerted': False, 'last_alert': None, 'samples': 0}
                self.states[kind] = state
            state.pop('clear_since', None)
            state['last'] = now
            state['samples'] += 1
            score = max(d['score'] for d in evidence)
            duration = now - state['start']
            urgent = score >= self.immediate
            sustained = state['samples'] >= 2 and duration >= self.persistence
            if (urgent or sustained) and (state['last_alert'] is None or now - state['last_alert'] >= self.cooldown):
                state['alerted'] = True
                state['last_alert'] = now
                name = '火焰' if kind == 'fire' else '烟雾'
                advice = '请立即确认现场情况，优先确保人员安全。'
                notices.append({
                    'id': str(uuid4()), 'episode_id': state['id'], 'type': kind,
                    'location': self.location, 'started_at': iso(state['start']),
                    'detected_at': iso(now), 'duration_seconds': round(duration, 1),
                    'decision': 'alert', 'confirmation': 'pending',
                    'reason': 'high_confidence' if urgent else 'persistent_detection',
                    'evidence': evidence, 'advice': advice,
                    'message': f'{self.location}检测到疑似{name}，已观测约{duration:.0f}秒。{advice}',
                })
        return notices

    def snapshot(self, now):
        return [{'type': kind, 'episode_id': s['id'], 'started_at': iso(s['start']),
                 'last_seen': iso(s['last']), 'duration_seconds': round(s['last'] - s['start'], 1),
                 'decision': 'alert' if s['alerted'] else 'observe',
                 'stale': now - s['last'] > self.max_gap,
                 'clearing': 'clear_since' in s}
                for kind, s in self.states.items()]
