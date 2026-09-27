"""User-requested JoyAI restart to load gradual sink decay (default off, then enabled live)."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
helper = ROOT / 'tulip/native600_trial.py'
ns = {'__file__': str(helper), '__name__': 'sink_decay_restart_bootstrap'}
exec(compile(helper.read_text().split('from easel_agent.ops_guard import hold_status')[0], str(helper), 'exec'), ns)
from easel_agent.ops_guard import hold_status

HALF_LIFE_S = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
if hold_status().get('held'):
    raise RuntimeError('Another operations hold is active')
code = ['deploy/xvideo/serving/cache_decay.py', 'deploy/xvideo/serving/joyomni_streaming.py',
        'deploy/xvideo/serving/serve_joyomni_streaming.py', 'tulip/serve.py', 'tulip/cache-decay.json']
out = ROOT / 'tulip/evidence' / time.strftime('sink-decay-restart-%Y%m%d-%H%M%S')
out.mkdir(exist_ok=False)
ns['out'] = out
current, act, save, stop, launch = [ns[k] for k in ('current', 'act', 'save', 'stop', 'launch')]
request = ns['adapter']._request
names = ('native-quality.json', 'interpolation-settings.json', 'output-presentation.json',
         'performance-settings.json', 'neural-rendering.json', 'stage-profile.json', 'native-capture.json',
         'shader-strength.json')
before = {name: json.loads((ROOT / 'tulip' / name).read_text(encoding='utf-8-sig')) for name in names}
save('before-settings', before)
save('code-hashes', {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in code})
for endpoint in ('health', 'debug'):
    save('before-' + endpoint, request('GET', 'http://127.0.0.1:7873/' + endpoint))
act('ops.hold', {'ttlMinutes': 20, 'reason': 'User requested JoyAI restart to load gradual sink decay',
                 'source': 'user-requested-sink-decay-restart'})
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
    save('captured-creative', {'prompt': prompt, 'strength': strength})
    stop()
    launch(before['native-quality.json'], prompt, strength, 'sink-decay')
    decay = request('GET', 'http://127.0.0.1:7873/cache-decay')
    save('decay-loaded', decay)
    if not decay.get('ok'):
        raise RuntimeError('cache-decay endpoint missing after restart')
    if HALF_LIFE_S > 0:
        save('decay-enabled', request('POST', 'http://127.0.0.1:7873/cache-decay',
                                      {'half_life_s': HALF_LIFE_S, 'mode': 'blend'}))
    for endpoint in ('health', 'debug'):
        save('after-' + endpoint, request('GET', 'http://127.0.0.1:7873/' + endpoint))
    save('after-live', current()[0])
    after = {name: json.loads((ROOT / 'tulip' / name).read_text(encoding='utf-8-sig')) for name in names}
    save('settings-preserved', {name: before[name] == after[name] for name in names})
    health = request('GET', 'http://127.0.0.1:7873/health')
    gpu = health['upscale']['gpu_video']
    print(json.dumps({'stage': 'sink-decay-live', 'evidence': str(out), 'geometry': gpu['geometry'],
                      'gpuError': gpu['error'], 'settingsPreserved': before == after,
                      'decay': request('GET', 'http://127.0.0.1:7873/cache-decay')['decay']}), flush=True)
finally:
    act('ops.resume', {'source': 'user-requested-sink-decay-restart'})
