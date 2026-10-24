import unittest

from xvideo.serving.feeder_trace import FeederTrace


class FeederTraceTests(unittest.TestCase):
    def test_blocked_feeder_is_observable_without_waiting_for_completion(self):
        now = [1.]
        trace = FeederTrace(clock=lambda: now[0], wall=lambda: 100.)
        with trace.measure('interop', queued=2, transition=True):
            now[0] += .3
            snapshot = trace.snapshot()
            self.assertEqual(snapshot['current']['stage'], 'interop')
            self.assertAlmostEqual(snapshot['current']['age_ms'], 300)
            self.assertEqual(snapshot['events'], [])
        row = trace.snapshot()['events'][0]
        self.assertEqual((row['queued'], row['transition'], row['at']), (2, True, 100.))
        self.assertAlmostEqual(row['duration_ms'], 300)
        self.assertIsNone(trace.snapshot()['current'])

    def test_history_is_bounded_and_exception_preserves_the_wait(self):
        now = [0.]
        trace = FeederTrace(capacity=2, clock=lambda: now[0])
        for _ in range(3):
            with self.assertRaisesRegex(ValueError, 'failed'):
                with trace.measure('select'):
                    now[0] += .1
                    raise ValueError('failed')
        snap = trace.snapshot()
        self.assertEqual([r['serial'] for r in snap['events']], [2, 3])
        self.assertEqual(snap['totals']['select']['count'], 3)
        self.assertIsNone(snap['current'])

    def test_fast_work_does_not_grow_history_or_mutate_old_snapshot(self):
        now = [0.]
        trace = FeederTrace(clock=lambda: now[0])
        before = trace.snapshot()
        for _ in range(100):
            with trace.measure('pack'):
                now[0] += .001
        self.assertEqual(trace.snapshot()['events'], [])
        self.assertEqual(trace.snapshot()['totals']['pack']['count'], 100)
        self.assertEqual(before['totals'], {})

    def test_completed_native_details_are_copied_into_the_matching_span(self):
        now = [0.]
        trace = FeederTrace(clock=lambda: now[0])
        with trace.measure('interop') as details:
            details.update(token=7, map_ms=80.)
            now[0] += .09
        details['token'] = 8
        row = trace.snapshot()['events'][0]
        self.assertEqual(row['details'], dict(token=7, map_ms=80.))
        self.assertAlmostEqual(row['duration_ms'], 90.)


if __name__ == '__main__':
    unittest.main()
