import unittest
from xvideo.serving.cadence_metrics import interval_metrics, require_native_cadence


class CadenceMetrics(unittest.TestCase):
    def test_bursts_with_good_average_are_rejected(self):
        gaps = [0, 1000/12]*100
        summary = interval_metrics(gaps, 1000/24)
        self.assertAlmostEqual(summary["meanMs"], 1000/24)
        self.assertEqual(summary["lateIntervals"], 100)
        state = dict(sent=len(gaps)+1, gap_sum_ms=sum(gaps),
                     gap_square_sum=sum(v*v for v in gaps), max_gap_ms=max(gaps),
                     underruns=0, missed_ticks=0)
        with self.assertRaises(AssertionError):
            require_native_cadence(state, 24)

    def test_native_steady_clock_passes_and_underrun_fails(self):
        state = dict(sent=241, gap_sum_ms=10000, gap_square_sum=240*(1000/24)**2,
                     max_gap_ms=43, underruns=0, missed_ticks=0)
        self.assertAlmostEqual(require_native_cadence(state, 24)["fps"], 24)
        state["underruns"] = 1
        with self.assertRaises(AssertionError):
            require_native_cadence(state, 24)

    def test_invalid_or_absent_measurement_is_not_success(self):
        for samples in ([], [float("nan")], [-1]):
            with self.assertRaises(ValueError):
                interval_metrics(samples, 1000/24)


if __name__ == "__main__":
    unittest.main()
