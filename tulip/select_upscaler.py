"""Explicit live SR selection at the next chunk boundary; never automatic."""
import argparse
import json
from pathlib import Path
from urllib.request import urlopen

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('backend', choices=['realesrgan', 'vsr_ultra'])
args = parser.parse_args()
with urlopen('http://127.0.0.1:7873/health', timeout=5) as response:
    health = json.load(response)
if args.backend not in health['upscale'].get('prewarmed_backends', []):
    raise SystemExit('Requested backend is not prewarmed in the live process')
path = Path(__file__).resolve().parent / 'output-presentation.json'
cfg = json.loads(path.read_text(encoding='utf-8-sig'))
cfg['backend'] = args.backend
pending = path.with_suffix('.selection-pending.json')
pending.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
pending.replace(path)
print('Requested ' + args.backend + '; applies at the next output chunk')
