"""User-requested JoyAI restart after CUDA OOM (allocator fragmentation); preserves prompt, strength, settings, cache decay."""
import json, time
from pathlib import Path

ROOT = Path('C:/Users/Admin/src/joyai-video-edit')
helper = ROOT / 'tulip/native600_trial.py'
ns = {'__file__': str(helper), '__name__': 'oom_restart_bootstrap'}
exec(compile(helper.read_text().split('from easel_agent.ops_guard import hold_status')[0], str(helper), 'exec'), ns)
from easel_agent.ops_guard import hold_status

if hold_status().get('held'):
    raise RuntimeError('Another operations hold is active')
out = ROOT / 'tulip/evidence' / time.strftime('oom-restart-%Y%m%d-%H%M%S')
out.mkdir(exist_ok=False)
ns['out'] = out
current, act, save, stop, launch = [ns[k] for k in ('current', 'act', 'save', 'stop', 'launch')]
request = ns['adapter']._request
names = ('native-quality.json', 'interpolation-settings.json', 'output-presentation.json',
         'performance-settings.json', 'neural-rendering.json', 'stage-profile.json', 'native-capture.json',
         'shader-strength.json')
before = {name: json.loads((ROOT / 'tulip' / name).read_text(encoding='utf-8-sig')) for name in names}
save('before-settings', before)
decay_before = request('GET', 'http://127.0.0.1:7873/cache-decay')
save('before-decay', decay_before)
for endpoint in ('health', 'debug'):
    save('before-' + endpoint, request('GET', 'http://127.0.0.1:7873/' + endpoint))
act('ops.hold', {'ttlMinutes': 20, 'reason': 'User requested JoyAI restart after CUDA OOM',
                 'source': 'user-requested-oom-restart'})
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
    launch(before['native-quality.json'], prompt, strength, 'oom-restart')
    decay = request('GET', 'http://127.0.0.1:7873/cache-decay')
    save('decay-loaded', decay)
    print(json.dumps({'stage': 'decay-after-launch', 'decay': decay.get('decay')}), flush=True)
    for endpoint in ('health', 'debug'):
        save('after-' + endpoint, request('GET', 'http://127.0.0.1:7873/' + endpoint))
    save('after-live', current()[0])
    after = {name: json.loads((ROOT / 'tulip' / name).read_text(encoding='utf-8-sig')) for name in names}
    save('settings-preserved', {name: before[name] == after[name] for name in names})
    mem = request('GET', 'http://127.0.0.1:7873/debug')['memory']['devices']
    print(json.dumps({'stage': 'oom-restart-live', 'evidence': str(out), 'settingsPreserved': before == after,
                      'memory': [{k: round(d[k]) if isinstance(d[k], float) else d[k] for k in
                                  ('device', 'allocatedMiB', 'reservedMiB', 'inactiveSplitMiB', 'allocationRetries')}
                                 for d in mem]}), flush=True)
finally:
    act('ops.resume', {'source': 'user-requested-oom-restart'})
