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
        self.assertEqual(ShaderStrength(0).snapshot()['referenceGain'], .75)
        self.assertEqual(ShaderStrength(1).snapshot()['blend'], .005)

    def test_bounded_latest_wins(self):
        s = ShaderStrength()
        for i in range(1000): s.set(i / 999)
        self.assertEqual(s.snapshot()['revision'], 1000)
        self.assertEqual(s.snapshot()['value'], .5)
        self.assertEqual(s.advance()['value'], .625)
        for _ in range(3): result = s.advance()
        self.assertTrue(result['settled'])
        self.assertEqual(result['target'], 1)

    def test_returns_to_exact_neutral_without_cache_operation(self):
        s = ShaderStrength(1)
        s.set(.5)
        for _ in range(4): result = s.advance()
        self.assertEqual(result['blend'], 0)
        self.assertEqual(result['referenceGain'], 1)
        self.assertTrue(result['cachePreserved'])

    def test_bad_input_does_not_mutate(self):
        s = ShaderStrength()
        for value in [True, None, '0.5', -1, 2, float('nan'), float('inf')]:
            with self.assertRaises(ValueError): s.set(value)
        self.assertEqual(s.snapshot()['revision'], 0)


if __name__ == '__main__': unittest.main()
