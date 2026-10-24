import unittest
from xvideo.serving.prompt_timing import PromptTiming, FrameOrigin, blended_origin


class PromptTimingTests(unittest.TestCase):
    def test_revision_identity_survives_a_newer_queued_prompt(self):
        now = [1000.]
        t = PromptTiming('first', clock=lambda: now[0])
        initial = t.frame(0, 0)
        now[0] = 1100
        t.requested(1, 'second')
        t.stage(1, 'encode_started_ms')
        now[0] = 1200
        t.stage(1, 'encode_finished_ms')
        self.assertEqual(t.frame(0, 1).prompt_hash, initial.prompt_hash)
        t.applied(1, 2)
        second = t.frame(1, 2)
        t.requested(2, 'third')
        self.assertEqual(t.frame(1, 3).prompt_hash, second.prompt_hash)
        self.assertEqual(second.requested_ms, 1100)
        self.assertEqual(second.encode_finished_ms, 1200)
        self.assertEqual(initial.revision, 0)
        self.assertIsNone(t.frame(2, 3))

    def test_restart_identity_and_same_text_updates_are_distinct(self):
        a, b = PromptTiming('same'), PromptTiming('same')
        self.assertNotEqual(a.session, b.session)
        a.frame(0, 0)
        a.requested(1, 'same')
        a.applied(1, 3)
        self.assertEqual(a.frame(1, 3).revision, 1)

    def test_bounded_history_never_relabels_evicted_work(self):
        t = PromptTiming('first', capacity=2)
        for i in range(1, 50):
            t.requested(i, str(i))
        self.assertEqual(len(t.requests), 2)
        t.applied(1, 5)
        self.assertIsNone(t.frame(1, 5))

    def test_transition_does_not_count_shader_as_new_generation(self):
        old = FrameOrigin('old', 0, 'a', 1, 100, 200)
        new = FrameOrigin('new', 0, 'b', 0, 300, 400)
        self.assertIsNone(blended_origin(old, None, 1))
        self.assertIsNone(blended_origin(None, new, 0))
        self.assertEqual(blended_origin(None, new, .1).generated_fraction, .1)
        self.assertEqual(blended_origin(old, None, .75).generated_fraction, .25)
        self.assertIs(blended_origin(None, new, 1), new)
        self.assertEqual(new.generated_fraction, 1)


if __name__ == '__main__':
    unittest.main()
