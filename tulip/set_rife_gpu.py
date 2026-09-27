"""Explicit, atomic live handoff toggle. Does not restart or change resolution."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('mode', choices=('on', 'off'))
args = parser.parse_args()
path = Path(__file__).resolve().parent / 'output-presentation.json'
config = json.loads(path.read_text(encoding='utf-8-sig'))
config['rife_gpu'] = args.mode == 'on'
pending = path.with_suffix('.rife-pending.json')
pending.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
pending.replace(path)
print(json.dumps({'rife_gpu': config['rife_gpu'], 'restartRequired': False,
                  'effective': 'after bounded in-flight chunks finish'}))
