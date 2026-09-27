"""Explicitly deploy UHD GPU output, preserving native generation and creative state."""
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
helper = ROOT / 'tulip/native600_trial.py'
ns = {'__file__': str(helper), '__name__': 'uhd_live_bootstrap'}
exec(compile(helper.read_text().split('from easel_agent.ops_guard import hold_status')[0], str(helper), 'exec'), ns)
from easel_agent.ops_guard import hold_status

if hold_status().get('held'):
    raise RuntimeError('Another operations hold is active')
out = ROOT / 'tulip/evidence/4k-live-20260924'
if not out.is_dir() or (out / 'before-live.json').exists():
    raise RuntimeError('Expected a fresh prepared evidence directory')
ns['out'] = out
current, act, save, stop, launch = [ns[k] for k in ('current', 'act', 'save', 'stop', 'launch')]
cfg = json.loads(ns['config_path'].read_text(encoding='utf-8-sig'))
assert cfg == {'width': 1152, 'height': 648, 'fps': 12, 'rife_multiplier': 2}
assert json.loads((ROOT / 'tulip/interpolation-settings.json').read_text())['backend'] == 'rife'
presentation_path = ROOT / 'tulip/output-presentation.json'
before = {name: json.loads((ROOT / 'tulip' / name).read_text(encoding='utf-8-sig'))
          for name in ('native-quality.json', 'interpolation-settings.json', 'output-presentation.json', 'performance-settings.json')}
presentation = before['output-presentation.json']
assert presentation['backend'] == 'vsr_ultra' and presentation['rife_gpu']
assert presentation['gpu_video']['enabled'] and presentation['gpu_video']['height'] == 1440
assert presentation['gpu_video']['bitrate'] == 15000000
save('before-settings', before)
for endpoint in ('health', 'debug'):
    save('before-' + endpoint, ns['adapter']._request('GET', 'http://127.0.0.1:7873/' + endpoint))
save('deployed-code', {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                       for name in ('deploy/xvideo/serving/upscale_output.py', 'deploy/xvideo/serving/gpu_video_output.py')})
act('ops.hold', {'ttlMinutes': 20, 'reason': 'User requested live 4K: load exact UHD output support and preserve 648p RIFE preset'})
try:
    listing, tile = current()
    if not tile or tile.get('status') != 'live':
        raise RuntimeError('Expected live JoyAI before deployment')
    save('before-live', listing)
    stats = tile['runtime']['stats']
    prompt = stats['prompt']
    strength = stats['strength'].get('target', stats['strength']['value'])
    # Reject concurrent settings edits before stopping the existing session.
    assert json.loads(presentation_path.read_text(encoding='utf-8-sig')) == presentation
    stop()
    presentation = json.loads(json.dumps(presentation))
    presentation['gpu_video']['height'] = 2160
    pending = presentation_path.with_suffix('.uhd-pending.json')
    pending.write_text(json.dumps(presentation, indent=2) + '\n', encoding='utf-8')
    pending.replace(presentation_path)
    launch(cfg, prompt, strength, 'uhd')
    time.sleep(5)
    for endpoint in ('health', 'debug'):
        save('after-' + endpoint, ns['adapter']._request('GET', 'http://127.0.0.1:7873/' + endpoint))
    save('after-live', current()[0])
    after = {name: json.loads((ROOT / 'tulip' / name).read_text(encoding='utf-8-sig')) for name in before}
    save('after-settings', after)
    assert after['output-presentation.json'] == presentation
    assert all(after[name] == before[name] for name in before if name != 'output-presentation.json')
    print(json.dumps({'stage': 'uhd-deployed', 'gpu_output': [3840, 2160, 24], 'bitrate': 15000000}), flush=True)
finally:
    # No timed or automatic quality rollback. Leave the selected 4K configuration in place.
    act('ops.resume')
