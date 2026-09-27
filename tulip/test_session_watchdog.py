import unittest
from xvideo.serving.session_watchdog import SessionWatchdog

class SessionWatchdogTests(unittest.TestCase):
    def test_disconnected_input_is_released_after_ten_seconds(self):
        w=SessionWatchdog()
        self.assertIsNone(w.check(now=9,last_activity=0,pending=False,progress=()))
        self.assertEqual(w.check(now=10,last_activity=0,pending=False,progress=()),'idle')

    def test_second_chunk_warmup_is_not_idle_after_first_output(self):
        w=SessionWatchdog()
        for now in (0,11,60,119):
            self.assertIsNone(w.check(now=now,last_activity=0,pending=True,progress=(1,1,1)))
        self.assertEqual(w.check(now=120,last_activity=0,pending=True,progress=(1,1,1)),'stalled')

    def test_new_inputs_cannot_hide_stalled_processing(self):
        w=SessionWatchdog()
        w.check(now=0,last_activity=0,pending=True,progress=(1,))
        self.assertEqual(w.check(now=120,last_activity=120,pending=True,progress=(1,)),'stalled')

    def test_worker_progress_extends_processing_deadline(self):
        w=SessionWatchdog()
        w.check(now=0,last_activity=0,pending=True,progress=(1,))
        self.assertIsNone(w.check(now=110,last_activity=0,pending=True,progress=(2,)))
        self.assertIsNone(w.check(now=229,last_activity=0,pending=True,progress=(2,)))
        self.assertEqual(w.check(now=230,last_activity=0,pending=True,progress=(2,)),'stalled')

    def test_rife_or_upscaling_counts_as_pending_work(self):
        w=SessionWatchdog()
        w.check(now=0,last_activity=0,pending=True,progress=(1,))
        self.assertIsNone(w.check(now=40,last_activity=0,pending=True,progress=(1,)))
        self.assertIsNone(w.check(now=50,last_activity=0,pending=False,progress=(2,)))
        self.assertEqual(w.check(now=60,last_activity=0,pending=False,progress=(2,)),'idle')

    def test_new_session_or_work_gets_its_own_deadline(self):
        w=SessionWatchdog()
        w.check(now=0,last_activity=0,pending=True,progress=(1,))
        w.check(now=1,last_activity=1,pending=False,progress=(2,))
        self.assertIsNone(w.check(now=100,last_activity=100,pending=True,progress=(2,)))
        self.assertIsNone(w.check(now=200,last_activity=100,pending=True,progress=(2,)))

if __name__=='__main__': unittest.main()
