import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('shader_strength', Path(__file__).resolve().parents[1] / 'deploy/xvideo/serving/shader_strength.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ShaderStrength = module.ShaderStrength


class ShaderDialTests(unittest.TestCase):
    def test_neutral_and_limits(self):
        s = ShaderStrength()
        self.assertEqual(s.snapshot()['blend'], 0)
        self.assertEqual(s.snapshot()['referenceGain'], 1)
        self.assertEqual(ShaderStrength(0).snapshot()['referenceGain'], 1)
        self.assertEqual(ShaderStrength(1).snapshot()['blend'], 1)
        self.assertEqual(s.snapshot()['neutral'], 0)

    def test_flux_units_are_direct_and_reference_conditioning_is_unchanged(self):
        for value in [0, .01, .03, .1, .13, .3, .5, .99, 1]:
            state = ShaderStrength(value).snapshot()
            self.assertEqual(state['blend'], value)
            self.assertEqual(state['referenceGain'], 1)
            self.assertEqual(state['mode'], 'direct_latent_blend')

    def test_bounded_latest_wins(self):
        s = ShaderStrength()
        for i in range(1000): s.set(i / 999)
        self.assertEqual(s.snapshot()['revision'], 1000)
        self.assertEqual(s.snapshot()['value'], 0)
        self.assertEqual(s.advance()['value'], .125)
        for _ in range(7): result = s.advance()
        self.assertTrue(result['settled'])
        self.assertEqual(result['target'], 1)

    def test_returns_to_exact_neutral_without_cache_operation(self):
        s = ShaderStrength(1)
        s.set(0)
        for _ in range(8): result = s.advance()
        self.assertEqual(result['blend'], 0)
        self.assertEqual(result['referenceGain'], 1)
        self.assertTrue(result['cachePreserved'])

    def test_bad_input_does_not_mutate(self):
        s = ShaderStrength()
        for value in [True, None, '0.5', -1, 2, float('nan'), float('inf')]:
            with self.assertRaises(ValueError): s.set(value)
        self.assertEqual(s.snapshot()['revision'], 0)


if __name__ == '__main__': unittest.main()
