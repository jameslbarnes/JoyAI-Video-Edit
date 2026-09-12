import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
from xvideo.serving.encoder_status import encoder_status


def pipeline(devices):
    return SimpleNamespace(text_execution_device="cuda:1", text_encoder=SimpleNamespace(
        parameters=lambda: [SimpleNamespace(device=d, dtype="torch.bfloat16", numel=lambda: 10) for d in devices]))


class EncoderStatusTests(unittest.TestCase):
    def test_unloaded(self):
        self.assertEqual(encoder_status(None), {"loaded": False})

    @patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,0"})
    def test_resident_and_physical_mapping(self):
        status = encoder_status(pipeline(["cuda:1", "cuda:1"]))
        self.assertTrue(status["fullyResident"])
        self.assertEqual(status["physicalGpu"], "0")
        self.assertEqual(status["parameterElementsByDevice"], {"cuda:1": 20})
        self.assertEqual(status["dtypes"], ["torch.bfloat16"])

    def test_offloaded_or_partially_resident(self):
        for devices in (["meta"], ["cuda:1", "cpu"], []):
            self.assertFalse(encoder_status(pipeline(devices))["fullyResident"])


if __name__ == "__main__":
    unittest.main()
