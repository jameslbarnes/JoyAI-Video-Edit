import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'deploy'))
from xvideo.serving.fresh_input import FreshInputTimeout, capture_model_input


class Reader:
    def __init__(self, outage=0, failure=None):
        self.now = 100.0
        self.ready_at = self.now+outage
        self.metrics = {}
        self.calls = []
        self.frame = object()
        self.failure = failure

    def capture(self, width, height, timeout):
        self.calls.append((width, height, timeout))
        if self.failure is not None:
            raise self.failure
        if self.now < self.ready_at:
            if self.now+timeout < self.ready_at:
                self.now += timeout
                raise FreshInputTimeout('source has not advanced')
            self.now = self.ready_at
        return self.frame


class FreshInputTests(unittest.TestCase):
    def capture(self, reader, **kwargs):
        with patch('xvideo.serving.fresh_input.time.monotonic', side_effect=lambda: reader.now):
            return capture_model_input(reader, 1152, 648, **kwargs)

    def test_816ms_source_interruption_preserves_only_the_next_fresh_frame(self):
        reader = Reader(.816)
        self.assertIs(self.capture(reader), reader.frame)
        self.assertEqual(len(reader.calls), 2)
        self.assertTrue(all(call[:2] == (1152,648) for call in reader.calls))
        self.assertEqual(reader.metrics['fresh_wait_timeouts'], 1)
        self.assertEqual(reader.metrics['fresh_wait_recoveries'], 1)
        self.assertAlmostEqual(reader.metrics['last_fresh_recovery_ms'], 816)

    def test_disconnected_source_fails_after_bounded_grace_without_stale_fallback(self):
        reader = Reader(100)
        with self.assertRaisesRegex(FreshInputTimeout, '2.0s grace'):
            self.capture(reader)
        self.assertEqual(reader.now, 102)
        self.assertEqual(len(reader.calls), 4)
        self.assertEqual(reader.metrics['fresh_wait_timeouts'], 4)
        self.assertNotIn('fresh_wait_recoveries', reader.metrics)

    def test_driver_failure_and_unrelated_timeout_are_not_retried(self):
        for error in (RuntimeError('input closed'), TimeoutError('driver failure')):
            reader = Reader(failure=error)
            with self.assertRaises(type(error)) as raised:
                self.capture(reader)
            self.assertIs(raised.exception, error)
            self.assertEqual(len(reader.calls), 1)
            self.assertEqual(reader.metrics, {'fresh_frame_grace_s': 2.0})

    def test_healthy_capture_has_no_retry_or_grace_delay(self):
        reader = Reader()
        self.assertIs(self.capture(reader), reader.frame)
        self.assertEqual(reader.now, 100)
        self.assertEqual(len(reader.calls), 1)
        self.assertEqual(reader.metrics, {'fresh_frame_grace_s': 2.0})

    def test_retry_uses_remaining_budget_and_validates_bounds(self):
        reader = Reader(100)
        with self.assertRaises(FreshInputTimeout):
            self.capture(reader, grace_s=.75)
        self.assertEqual([x[2] for x in reader.calls], [.5,.25])
        for value in (0, 10):
            with self.assertRaises(ValueError):
                self.capture(Reader(), grace_s=value)


if __name__ == '__main__':
    unittest.main()
