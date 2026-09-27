import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
from xvideo.serving.interpolation_output import InterpolationOutput,selected_backend


class Dummy:
    def __init__(self):self.resets=[];self.closed=False
    def reset(self,**kwargs):self.resets.append(kwargs)
    def process(self,*args,**kwargs):return ['output'],[{'seq':1}]
    def close(self):self.closed=True


class BackendSelection(unittest.TestCase):
    def make(self):
        r=InterpolationOutput.__new__(InterpolationOutput)
        r.backend='rife';r.models={k:Dummy() for k in ('rife','dlssg')};r._transition=False
        r.capture=Mock()
        return r

    def test_explicit_switch_resets_history_once_and_retains_both_engines(self):
        r=self.make();profile={}
        for name in ('dlssg','dlssg','rife'):
            with patch('xvideo.serving.interpolation_output.selected_backend',return_value=name):
                r.process([1],[{}],profile)
            self.assertEqual(profile['interpolation_backend'],name)
        self.assertEqual(len(r.models['dlssg'].resets),1)
        self.assertEqual(len(r.models['rife'].resets),1)

    def test_scene_reset_clears_both_cached_histories(self):
        r=self.make();r.reset(transition=True)
        self.assertTrue(r.transition_pending)
        self.assertTrue(all(m.resets==[{'transition':True}] for m in r.models.values()))
        r.backend='dlssg';self.assertFalse(r.transition_pending)

    def test_dlssg_error_is_not_silently_replaced_by_rife(self):
        r=self.make()
        def fail(*args,**kwargs):raise RuntimeError('native failure')
        r.models['dlssg'].process=fail
        with patch('xvideo.serving.interpolation_output.selected_backend',return_value='dlssg'):
            with self.assertRaisesRegex(RuntimeError,'native failure'):r.process([1],[{}],{})
        self.assertEqual(r.models['rife'].resets,[])

    def test_config_is_explicit_and_defaults_to_rife(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'settings.json';self.assertEqual(selected_backend(p),'rife')
            p.write_text(json.dumps({'backend':'dlssg'}));self.assertEqual(selected_backend(p),'dlssg')
            p.write_text(json.dumps({'backend':'unknown'}))
            with self.assertRaises(ValueError):selected_backend(p)

    def test_close_releases_all_engines(self):
        r=self.make();models=list(r.models.values());r.close()
        self.assertFalse(r.models);self.assertTrue(all(m.closed for m in models))

if __name__=='__main__':unittest.main()
