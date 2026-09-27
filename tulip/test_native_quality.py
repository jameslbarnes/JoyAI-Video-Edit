import json
from pathlib import Path
import tempfile
import unittest
from xvideo.serving.native_config import load_native_config

class NativeQuality(unittest.TestCase):
    def test_bounded_profiles_and_rollback(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'native.json'
            self.assertEqual(load_native_config(path)['width'], 648)
            for width,height,fps,rife in [(840,480,12,4),(864,480,16,2),(768,432,20,2),(1056,600,12,2),(1152,648,12,2),(1296,720,12,2),(1296,720,6,4)]:
                profile = dict(width=width,height=height,fps=fps,rife_multiplier=rife)
                path.write_text(json.dumps(profile))
                self.assertEqual(load_native_config(path), profile)
            for invalid in [{'width':9000}, {'height':True}, {'fps':0},
                            {'fps':24,'rife_multiplier':4}, {'steps':1}, []]:
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError): load_native_config(path)
            path.unlink()
            self.assertEqual(load_native_config(path)['fps'],16)

if __name__ == '__main__': unittest.main()
