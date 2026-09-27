"""User-requested runtime restart for the tested cache-teardown GPU barrier."""
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
helper = ROOT / 'tulip/native600_trial.py'
ns = {'__file__': str(helper), '__name__': 'cache_fix_restart_bootstrap'}
exec(compile(helper.read_text().split('from easel_agent.ops_guard import hold_status')[0], str(helper), 'exec'), ns)
from easel_agent.ops_guard import hold_status

if hold_status().get('held'):
    raise RuntimeError('Another operations hold is active')
prepared = json.loads((ROOT / 'tulip/evidence/cache-audit-20260924-185144/prepared-fix.json').read_text())
for name, expected in prepared['files'].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, f'Tested source changed: {name}'
out = ROOT / 'tulip/evidence' / time.strftime('cache-fix-restart-%Y%m%d-%H%M%S')
out.mkdir(exist_ok=False)
(ROOT / 'tulip/cache-fix-restart-current.txt').write_text(str(out), encoding='utf-8')
ns['out'] = out
current, act, save, stop, launch = [ns[k] for k in ('current', 'act', 'save', 'stop', 'launch')]
names = ('native-quality.json', 'interpolation-settings.json', 'output-presentation.json',
         'performance-settings.json', 'neural-rendering.json', 'stage-profile.json', 'native-capture.json')
before = {name: json.loads((ROOT / 'tulip' / name).read_text(encoding='utf-8-sig')) for name in names}
cfg = before['native-quality.json']
assert cfg == {'width': 1152, 'height': 648, 'fps': 12, 'rife_multiplier': 2}
assert before['interpolation-settings.json']['backend'] == 'rife'
assert before['output-presentation.json']['gpu_video']['height'] == 2160
assert before['output-presentation.json']['gpu_video']['bitrate'] == 15000000
save('before-settings', before)
save('tested-code', prepared)
for endpoint in ('health', 'debug'):
    save('before-' + endpoint, ns['adapter']._request('GET', 'http://127.0.0.1:7873/' + endpoint))
save('before-runtime', ns['adapter']._request('GET', ns['backend'] + '/api/runtimes/joyai?dependencies=0'))
act('ops.hold', {'ttlMinutes': 20, 'reason': 'User requested JoyAI restart to load tested cache-teardown GPU drain',
                 'source': 'user-requested-cache-fix-restart'})
try:
    listing, tile = current()
    if not tile or tile.get('status') != 'live':
        raise RuntimeError('Expected live JoyAI tile before restart')
    save('before-live', listing)
    stats = tile['runtime']['stats']
    if stats.get('referenceImageActive'):
        raise RuntimeError('Current reference image needs preserving before session recreation')
    prompt = stats['prompt']
    strength = stats['strength'].get('target', stats['strength']['value'])
    # Capture creative state immediately before teardown; launch preserves any newer edits.
    assert all(json.loads((ROOT / 'tulip' / name).read_text(encoding='utf-8-sig')) == before[name] for name in names)
    stop()
    launch(cfg, prompt, strength, 'cache-fix')
    for endpoint in ('health', 'debug'):
        save('after-' + endpoint, ns['adapter']._request('GET', 'http://127.0.0.1:7873/' + endpoint))
    save('after-live', current()[0])
    after = {name: json.loads((ROOT / 'tulip' / name).read_text(encoding='utf-8-sig')) for name in names}
    save('after-settings', after)
    save('settings-preserved', {name: before[name] == after[name] for name in names})
    assert before == after, 'A setting changed concurrently; inspect saved states without overwriting it'
    health = ns['adapter']._request('GET', 'http://127.0.0.1:7873/health')
    gpu = health['upscale']['gpu_video']
    assert health['upscale']['enabled'] and not health['upscale']['error']
    assert gpu['geometry'] == [3840, 2160, 24] and not gpu['error']
    print(json.dumps({'stage': 'cache-fix-live', 'evidence': str(out), 'geometry': gpu['geometry'],
                      'bitrate': gpu['bitrate'], 'settingsPreserved': before == after}), flush=True)
finally:
    act('ops.resume', {'source': 'user-requested-cache-fix-restart'})
