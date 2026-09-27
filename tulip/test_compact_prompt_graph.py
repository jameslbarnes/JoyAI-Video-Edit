"""Compact graph capacities must cover prompts without truncating their prefix."""
import os
import unittest
from unittest.mock import patch
from xvideo.serving.graph_runner import prompt_graph_capacity, prompt_graph_capacities

class CompactPromptGraphs(unittest.TestCase):
    def test_single_graph_keeps_every_supported_prompt_without_truncation(self):
        with patch.dict(os.environ, JOYAI_SINGLE_PROMPT_GRAPH='1'):
            self.assertEqual(prompt_graph_capacities(1024),(1024,))
            for length in range(1,1025):
                self.assertEqual(prompt_graph_capacity(length,1024),1024)
            self.assertEqual(prompt_graph_capacities(768),(768,))
            with self.assertRaises(ValueError): prompt_graph_capacity(1025,1024)
    def test_every_supported_prompt_fits_a_prewarmed_bucket(self):
        with patch.dict(os.environ, JOYAI_COMPACT_PROMPT_GRAPHS='1'):
            self.assertEqual(prompt_graph_capacities(1024), (512,1024))
            for length in range(1,1025):
                capacity=prompt_graph_capacity(length,1024)
                self.assertGreaterEqual(capacity,length)
                self.assertIn(capacity,(512,1024))
            self.assertEqual(prompt_graph_capacity(512,1024),512)
            self.assertEqual(prompt_graph_capacity(513,1024),1024)
            for invalid in (0,1025):
                with self.assertRaises(ValueError): prompt_graph_capacity(invalid,1024)

    def test_default_and_smaller_limits_remain_bounded(self):
        with patch.dict(os.environ, JOYAI_COMPACT_PROMPT_GRAPHS='0'):
            self.assertEqual(prompt_graph_capacities(1024),(512,640,768,896,1024))
        with patch.dict(os.environ, JOYAI_COMPACT_PROMPT_GRAPHS='1'):
            self.assertEqual(prompt_graph_capacities(768),(512,768))
            self.assertEqual(prompt_graph_capacity(513,768),768)

if __name__=='__main__': unittest.main()
