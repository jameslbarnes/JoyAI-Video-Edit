"""User-approved GPU RIFE integration using only the existing JoyAI lifecycle."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('evidence', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
helper = root / 'tulip/native600_trial.py'
ns = {'__file__': str(helper), '__name__': 'rife_gpu_bootstrap'}
exec(compile(helper.read_text().split('from easel_agent.ops_guard import hold_status')[0], str(helper), 'exec'), ns)
from easel_agent.ops_guard import hold_status
if hold_status().get('held'):
    raise RuntimeError('Another operations hold is active')
out = args.evidence.resolve()
if not out.is_dir(): raise RuntimeError('Create the evidence snapshot first')
ns['out'] = out
current, act, save, stop, launch = [ns[k] for k in ('current', 'act', 'save', 'stop', 'launch')]
listing, tile = current()
if not tile or tile.get('status') != 'live': raise RuntimeError('Expected the current live JoyAI tile')
save('deploy-before-live', listing)
stats = tile['runtime']['stats']
prompt = stats['prompt']
strength = stats['strength'].get('target', stats['strength']['value'])
cfg = json.loads(ns['config_path'].read_text(encoding='utf-8-sig'))
if cfg != {'width': 1152, 'height': 648, 'fps': 12, 'rife_multiplier': 2}:
    raise RuntimeError('Native profile changed; preserve it')
path = root / 'tulip/output-presentation.json'
presentation = json.loads(path.read_text(encoding='utf-8-sig'))
if presentation.get('backend') != 'vsr_ultra': raise RuntimeError('SR backend changed; preserve it')
save('deploy-before-presentation', presentation)
act('ops.hold', {'ttlMinutes': 30, 'reason': 'User-approved GPU-only RIFE handoff at existing 648p profile'})
try:
    stop()
    presentation['rife_gpu'] = True
    pending = path.with_suffix('.rife-pending.json')
    pending.write_text(json.dumps(presentation, indent=2) + '\n', encoding='utf-8')
    pending.replace(path)
    launch(cfg, prompt, strength, 'rife-gpu')
    for endpoint in ('health', 'debug'):
        save('deploy-after-' + endpoint, ns['adapter']._request('GET', 'http://127.0.0.1:7873/' + endpoint))
finally:
    act('ops.resume')
