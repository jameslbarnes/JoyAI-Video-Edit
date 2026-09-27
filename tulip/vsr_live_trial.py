"""User-authorized VSR Ultra trial, unchanged native quality, no automatic rollback."""
from pathlib import Path
import json
root = Path(__file__).resolve().parents[1]
h = root / 'tulip/native600_trial.py'
ns = {'__file__': str(h), '__name__': 'vsr_trial_bootstrap'}
exec(compile(h.read_text().split('from easel_agent.ops_guard import hold_status')[0], str(h), 'exec'), ns)
from easel_agent.ops_guard import hold_status
if hold_status().get('held'): raise RuntimeError('Another operations hold is active')
out = root / 'tulip/evidence/vsr-live-20260922'
out.mkdir(parents=True, exist_ok=True)
ns['out'] = out
current, act, save, stop, launch = [ns[k] for k in ('current','act','save','stop','launch')]
listing, tile = current()
if not tile or tile.get('status') != 'live': raise RuntimeError('Expected a live JoyAI tile')
save('before-live', listing)
save('before-health', ns['adapter']._request('GET', 'http://127.0.0.1:7873/health'))
stats = tile['runtime']['stats']
prompt = stats['prompt']
strength = stats['strength'].get('target', stats['strength']['value'])
cfg = json.loads(ns['config_path'].read_text())
if cfg != {'width':1152,'height':648,'fps':12,'rife_multiplier':2}:
    raise RuntimeError('Native profile changed; preserve user selection')
path = root / 'tulip/output-presentation.json'
presentation = json.loads(path.read_text(encoding='utf-8-sig'))
save('before-presentation', presentation)
act('ops.hold', {'ttlMinutes':25, 'reason':'Explicit VSR Ultra live trial at unchanged 648p/RIFE 2; no automatic rollback'})
try:
    stop()
    presentation.update(backend='vsr_ultra', preload_vsr=True)
    pending = path.with_suffix('.vsr-pending.json')
    pending.write_text(json.dumps(presentation, indent=2), encoding='utf-8')
    pending.replace(path)
    launch(cfg, prompt, strength, 'native648-vsr')
    health = ns['adapter']._request('GET', 'http://127.0.0.1:7873/health')
    save('after-health', health)
    if health['upscale'].get('backend') != 'vsr_ultra' or health['upscale'].get('error'):
        raise RuntimeError('VSR did not initialize successfully; runtime preserved for diagnosis')
finally:
    act('ops.resume')
