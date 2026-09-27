"""Explicit same-resolution VAE placement trial; never automatically rolls back."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('evidence', type=Path)
parser.add_argument('--placement', choices=('baseline', 'encode', 'pseudo', 'decode'), required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
helper = root / 'tulip/native600_trial.py'
ns = {'__file__': str(helper), '__name__': 'profile_vae_bootstrap'}
exec(compile(helper.read_text().split('from easel_agent.ops_guard import hold_status')[0], str(helper), 'exec'), ns)
from easel_agent.ops_guard import hold_status
if hold_status().get('held'):
    raise RuntimeError('Another operations hold is active')
out = args.evidence.resolve()
out.mkdir(parents=True, exist_ok=False)
ns['out'] = out
current, act, save, stop, launch = [ns[k] for k in ('current', 'act', 'save', 'stop', 'launch')]
listing, tile = current()
if not tile or tile.get('status') != 'live':
    raise RuntimeError('Expected the current live JoyAI tile')
save('before-live', listing)
stats = tile['runtime']['stats']
prompt = stats['prompt']
strength = stats['strength'].get('target', stats['strength']['value'])
cfg = json.loads(ns['config_path'].read_text(encoding='utf-8-sig'))
if cfg != {'width': 1152, 'height': 648, 'fps': 12, 'rife_multiplier': 2}:
    raise RuntimeError('648p profile changed; preserve it')
path = root / 'tulip/performance-settings.json'
performance = json.loads(path.read_text(encoding='utf-8-sig'))
save('before-performance', performance)
save('before-native', cfg)
for endpoint in ('health', 'debug'):
    save('before-' + endpoint, ns['adapter']._request('GET', 'http://127.0.0.1:7873/' + endpoint))
performance.update(vaeDevice='output' if args.placement == 'decode' else 'model',
                   vaeEncodeDevice='output' if args.placement == 'encode' else 'model',
                   vaePseudoDevice='output' if args.placement == 'pseudo' else 'model')
if args.placement == 'baseline':
    performance.pop('vaeEncodeDevice', None)
    performance.pop('vaePseudoDevice', None)
print(json.dumps({'placement': args.placement, 'native': cfg, 'automaticRollback': False}), flush=True)
act('ops.hold', {'ttlMinutes': 30, 'reason': 'User-approved 648p VAE stage profiling: ' + args.placement})
try:
    stop()
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(performance, indent=2) + '\n', encoding='utf-8')
    pending.replace(path)
    profile_path = root / 'tulip/stage-profile.json'
    pending = profile_path.with_suffix('.pending.json')
    pending.write_text('{"enabled": true}\n', encoding='utf-8')
    pending.replace(profile_path)
    launch(cfg, prompt, strength, 'vae-' + args.placement)
    for endpoint in ('health', 'debug'):
        save('after-' + endpoint, ns['adapter']._request('GET', 'http://127.0.0.1:7873/' + endpoint))
finally:
    act('ops.resume')
