import unittest

from app.fire.rules import FireRules


def detection(label='fire', score=0.9):
    return [{'label': label, 'score': score, 'bbox': [1, 2, 20, 30]}]


class FireRulesTests(unittest.TestCase):
    def test_obvious_fire_and_smoke_alert_immediately(self):
        rules = FireRules('厨房')
        events = rules.update(detection() + detection('smoke'), 100)
        self.assertEqual([e['type'] for e in events], ['fire', 'smoke'])
        self.assertEqual(events[0]['duration_seconds'], 0)
        self.assertEqual(events[0]['location'], '厨房')
        self.assertEqual(events[0]['confirmation'], 'pending')

    def test_weak_candidate_requires_persistence(self):
        rules = FireRules('厨房')
        self.assertEqual(rules.update(detection(score=.5), 100), [])
        self.assertEqual(rules.snapshot(100)[0]['decision'], 'observe')
        self.assertEqual(rules.update(detection(score=.5), 104), [])
        self.assertEqual(rules.update(detection(score=.5), 105)[0]['reason'], 'persistent_detection')

    def test_weak_interrupted_candidate_does_not_accumulate(self):
        rules = FireRules('厨房')
        rules.update(detection(score=.5), 100)
        rules.update([], 104)
        self.assertEqual(rules.update(detection(score=.5), 105), [])

    def test_low_scores_and_unrelated_labels_ignored(self):
        rules = FireRules('厨房')
        self.assertEqual(rules.update(detection(score=.39) + detection('person'), 100), [])
        self.assertEqual(rules.snapshot(100), [])

    def test_repeat_cooldown_and_independent_smoke(self):
        rules = FireRules('厨房')
        first = rules.update(detection(), 100)[0]
        for now in range(101, 160):
            self.assertEqual(rules.update(detection(), now), [])
        repeat = rules.update(detection() + detection('smoke'), 160)
        self.assertEqual(len(repeat), 2)
        self.assertEqual(repeat[0]['episode_id'], first['episode_id'])
        self.assertNotEqual(repeat[0]['id'], first['id'])

    def test_recovery_and_new_episode(self):
        rules = FireRules('厨房')
        first = rules.update(detection(), 100)[0]
        rules.update([], 101)
        rules.update([], 110)
        rules.update([], 111)
        self.assertEqual(rules.snapshot(111), [])
        second = rules.update(detection(), 112)[0]
        self.assertNotEqual(first['episode_id'], second['episode_id'])

    def test_camera_gap_is_not_continuous_evidence(self):
        rules = FireRules('厨房')
        rules.update(detection(score=.5), 100)
        self.assertTrue(rules.snapshot(111)[0]['stale'])
        self.assertEqual(rules.update(detection(score=.5), 120), [])


if __name__ == '__main__':
    unittest.main()
