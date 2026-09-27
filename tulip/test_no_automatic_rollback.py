"""Failure-reporting paths must never stop or restore the live model."""
import ast
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT=Path(__file__).resolve().parent
class NoAutomaticRollbackTests(unittest.TestCase):
    def run_failure(self,name,stage):
        tree=ast.parse((ROOT/name).read_text(encoding='utf-8-sig'))
        body=[n for n in tree.body if isinstance(n,ast.Try)][-1]
        stop=Mock(); launch=Mock(); switch=Mock();observe=Mock()
        fail=RuntimeError('injected quality check failure')
        {'launch':launch,'switch':switch,'observe':observe}[stage].side_effect=fail
        ns={'stop':stop,'launch':launch,'switch':switch,'observe':observe,'act':Mock(),'save':Mock(),
            'rollback':False,'prompt':'current','strength':0.2,'prior':{'width':840},
            'json':__import__('json'),'print':Mock(),'cfg':{'width':1152}}
        with self.assertRaises(RuntimeError):exec(compile(ast.Module(body=[body],type_ignores=[]),name,'exec'),ns)
        self.assertLessEqual(stop.call_count,1)
        self.assertLessEqual(launch.call_count,1)
        self.assertLessEqual(switch.call_count,1)
        if launch.call_args:self.assertIn(launch.call_args.args[0]['width'],(1056,1152))
        if switch.call_args:self.assertEqual(switch.call_args.args[0]['width'],1056)
        ns['act'].assert_called_once_with('ops.resume')
    def test_initialization_failure_does_not_restore(self):self.run_failure('native600_trial.py','launch')
    def test_session_failure_does_not_restore(self):self.run_failure('native600_retry.py','switch')
    def test_quality_check_failure_keeps_rendering(self):self.run_failure('native600_guard_trial.py','observe')
    def test_648_start_failure_does_not_restore(self):self.run_failure('native648_trial.py','launch')
    def test_guarded_start_failure_does_not_restore(self):self.run_failure('native600_guard_trial.py','launch')

if __name__=='__main__':unittest.main()
