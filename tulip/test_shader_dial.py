import importlib.util
from pathlib import Path
import unittest
import tempfile
import torch

spec = importlib.util.spec_from_file_location('shader_strength', Path(__file__).resolve().parents[1] / 'deploy/xvideo/serving/shader_strength.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ShaderStrength = module.ShaderStrength


class ShaderDialTests(unittest.TestCase):
    def test_combined_curve_endpoints_baseline_and_monotonicity(self):
        self.assertEqual(module.strength_inputs('combined',0),(0,0,0))
        self.assertEqual(module.strength_inputs('combined',.6),(0,1,0))
        self.assertEqual(module.strength_inputs('combined',1),(.1,1,1))
        previous=(0,0,0)
        for i in range(101):
            state=module.strength_inputs('combined',i/100)
            self.assertTrue(all(a<=b for a,b in zip(previous,state)))
            previous=state
        self.assertLess(module.strength_inputs('combined',.75)[0],.005)
        self.assertEqual(module.strength_inputs('combined',.9)[2],0)

    def test_combined_ramps_both_ways_and_equivalent_mode_rollback(self):
        dial=ShaderStrength(0,mode='combined')
        for target in [1,.6,0,1,0]:
            dial.set(target)
            previous=dial.snapshot()
            for _ in range(24):
                state=dial.advance()
                for key in ['value','blend','referenceGain']:
                    self.assertLessEqual(abs(state[key]-previous[key]),.125+1e-10)
                self.assertEqual(state['outputMixPrevious'],previous['outputMix'])
                previous=state
            self.assertTrue(state['settled'])
            self.assertEqual((state['blend'],state['referenceGain'],state['outputMix']),module.strength_inputs('combined',target))
        dial=ShaderStrength(.6,mode='combined')
        for mode,value in [('reference_only',1),('direct_latent_blend',0),('combined',.6)]:
            dial.set(value,mode=mode)
            for _ in range(12):
                state=dial.advance()
                self.assertEqual((state['blend'],state['referenceGain'],state['outputMix']),(0,1,0))

    def test_display_mix_exact_endpoints_and_frame_ramp_do_not_mutate_history(self):
        generated=torch.zeros((8,4,4,3),dtype=torch.uint8)
        source=torch.full_like(generated,200)
        self.assertIs(module.mix_shader_presentation(generated,None,{}),generated)
        self.assertIs(module.mix_shader_presentation(generated,source,{'outputMix':1,'outputMixPrevious':1}),source)
        mixed=module.mix_shader_presentation(generated,source,{'outputMix':1,'outputMixPrevious':0})
        self.assertEqual(mixed[:,0,0,0].tolist(),[25,50,75,100,125,150,175,200])
        mixed=module.mix_shader_presentation(generated,source,{'outputMix':0,'outputMixPrevious':1})
        self.assertEqual(mixed[:,0,0,0].tolist(),[175,150,125,100,75,50,25,0])
        self.assertEqual(torch.count_nonzero(generated).item(),0)
        self.assertTrue(torch.all(source==200))
        with self.assertRaises(ValueError):module.mix_shader_presentation(generated,None,{'outputMix':1})
        with self.assertRaises(ValueError):module.mix_shader_presentation(generated,source[:1],{'outputMix':1})

    def test_reference_mode_controls_only_reference_in_both_directions(self):
        dial = ShaderStrength(1, mode='reference_only')
        for target in [0, .25, .75, 1, .5, 0]:
            before = dial.snapshot()
            dial.set(target)
            for _ in range(8):
                state = dial.advance()
                self.assertEqual(state['blend'], 0)
                self.assertEqual(state['referenceGain'], state['value'])
                self.assertLessEqual(abs(state['referenceGain']-before['referenceGain']), .125)
                before = state
            self.assertTrue(state['settled'])
            self.assertEqual(state['referenceGain'], target)

    def test_mode_switch_and_rollback_smooth_actual_inputs(self):
        dial = ShaderStrength(.25)
        original = dial.snapshot()
        queued = dial.set(0, mode='reference_only')
        self.assertEqual(queued['referenceGain'], original['referenceGain'])
        self.assertFalse(queued['settled'])
        for _ in range(8): dial.advance()
        self.assertEqual(dial.snapshot()['referenceGain'], 0)
        dial.set(.25, mode='direct_latent_blend')
        for _ in range(8): dial.advance()
        restored = dial.snapshot()
        self.assertEqual((restored['blend'], restored['referenceGain']), (.25, 1))
        self.assertTrue(restored['settled'])
        with self.assertRaises(ValueError): dial.set(.5, mode='unknown')
        self.assertEqual(dial.snapshot(), restored)

    def test_equivalent_baselines_do_not_inject_shader_during_mode_switch(self):
        dial = ShaderStrength(0)
        for mode, value in [('reference_only',1),('direct_latent_blend',0)]:
            dial.set(value, mode=mode)
            for _ in range(8):
                state = dial.advance()
                self.assertEqual(state['blend'], 0)
                self.assertEqual(state['referenceGain'], 1)
            self.assertTrue(state['settled'])

    def test_zero_is_independent_of_fresh_shader_content_without_mutation(self):
        noise = torch.randn(1, 4, 1, 4, 4)
        source = torch.randn_like(noise)
        saved = source.clone()
        zero = ShaderStrength(0, mode='reference_only').snapshot()
        a, ref_a = module.prepare_strength_inputs(noise, source, zero)
        b, ref_b = module.prepare_strength_inputs(noise, source + 100, zero)
        self.assertIs(a, noise)
        self.assertIs(b, noise)
        self.assertEqual(torch.count_nonzero(ref_a).item(), 0)
        torch.testing.assert_close(ref_a, ref_b, rtol=0, atol=0)
        torch.testing.assert_close(source, saved, rtol=0, atol=0)
        full = ShaderStrength(1, mode='reference_only').snapshot()
        generated, reference = module.prepare_strength_inputs(noise, source, full)
        self.assertIs(generated, noise)
        self.assertIs(reference, source)
        for gain in [.25, .5, .75]:
            _, reference = module.prepare_strength_inputs(noise, source,
                ShaderStrength(gain, mode='reference_only').snapshot())
            torch.testing.assert_close(reference, source * gain)

    def test_configuration_is_explicit_and_legacy_default_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'strength.json'
            self.assertEqual(module.load_strength_config(path), {'mode':'direct_latent_blend','value':0})
            path.write_text('{"mode":"reference_only","value":1}')
            self.assertEqual(ShaderStrength(**module.load_strength_config(path)).snapshot()['referenceGain'], 1)
            path.write_text('{"mode":"reference_only","value":2}')
            with self.assertRaises(ValueError): module.load_strength_config(path)

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
