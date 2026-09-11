import unittest

import torch

from strength_controls import prepare_strength_inputs, strength_for_chunk


class StrengthControls(unittest.TestCase):
    def setUp(self):
        self.noise = torch.tensor([[[[[2., -2.]]]]])
        self.ref = torch.tensor([[[[[-1., 1.]]]]])

    def test_neutral_is_exact_noop(self):
        noise, ref = prepare_strength_inputs(self.noise, self.ref)
        self.assertIs(noise, self.noise)
        self.assertIs(ref, self.ref)

    def test_blend_endpoints_and_midpoint_preserve_inputs(self):
        for blend in [0, .25, .5, 1]:
            noise, ref = prepare_strength_inputs(self.noise, self.ref, blend=blend)
            torch.testing.assert_close(noise, self.noise * (1-blend) + self.ref * blend)
            self.assertIs(ref, self.ref)
        torch.testing.assert_close(self.noise.flatten(), torch.tensor([2., -2.]))
        torch.testing.assert_close(self.ref.flatten(), torch.tensor([-1., 1.]))

    def test_reference_gain_does_not_change_starting_noise(self):
        noise, ref = prepare_strength_inputs(self.noise, self.ref, reference_gain=.5)
        self.assertIs(noise, self.noise)
        torch.testing.assert_close(ref, self.ref * .5)

    def test_reject_invalid_values_and_shapes(self):
        for value in [-1, 1.1, float('nan'), float('inf')]:
            for key in ['blend', 'reference_gain']:
                with self.assertRaises(ValueError):
                    prepare_strength_inputs(self.noise, self.ref, **{key: value})
        with self.assertRaises(ValueError):
            prepare_strength_inputs(self.noise, self.ref.flatten())
        with self.assertRaises(ValueError):
            prepare_strength_inputs(self.noise, self.ref.double())

    def test_transition_schedule_returns_to_neutral(self):
        self.assertEqual([strength_for_chunk('strength_ramp', i)[0] for i in [0,16,32,48,64]],
                         [0, .25, .5, 0, 0])


if __name__ == '__main__':
    unittest.main()
