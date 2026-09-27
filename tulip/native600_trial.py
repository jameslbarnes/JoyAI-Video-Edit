"""Bounded 600p live trial using the existing SDK/runtime lifecycle, without automatic rollback."""
import json
from pathlib import Path
import sys
import time

root = Path(__file__).resolve().parents[1]
helper = root / 'tulip/quality_trial_lifecycle.py'
ns = {'__file__': str(helper), '__name__': 'native600_bootstrap'}
exec(compile(helper.read_text().split('listing =')[0], str(helper), 'exec'), ns)
router, adapter = ns['router'], ns['adapter']
out = root / 'tulip/evidence/native600-20260922'
config_path = root / 'tulip/native-quality.json'
backend, control = ns['backend'], ns['control']
rollback = '--rollback' in sys.argv

def save(name, data):
    (out / (name + '.json')).write_text(json.dumps(data, indent=2), encoding='utf-8')

def act(name, payload=None):
    r = router.dispatch(name, {'config': str(ns['sdk'] / 'nanos.json'),
        'source': 'user-requested-native600-trial', **(payload or {})})
    save(name.replace('.', '-') + ('-rollback' if rollback else ''),
         {**r, 'data': {k: v for k, v in r.get('data', {}).items() if k != 'state'}})
    if not r.get('ok'): raise RuntimeError(f'{name}: {r.get("error")}')
    return r

def current():
    listing = adapter._request('GET', control + '/control/multistream/list')
    return listing, next((s for s in listing['streams'] if s.get('type') == 'joyai'), None)

def stop():
    if current()[1]: act('multistream.remove', {'provider': 'joyai', 'force': True})
    r = adapter._request('POST', backend + '/api/runtimes/joyai/stop',
                         {'force': True, 'cleanupOrphans': False}, timeout=40)
    if r.get('process_running'): raise RuntimeError('JoyAI did not stop cleanly')

def launch(config, prompt, strength, label):
    pending = config_path.with_suffix('.trial-pending.json')
    pending.write_text(json.dumps(config)); pending.replace(config_path)
    adapter._request('POST', backend + '/api/runtimes/joyai/start', {'wait': False}, timeout=20)
    started = time.monotonic(); last_print = 0
    while time.monotonic() - started < 900:
        r = adapter._request('GET', backend + '/api/runtimes/joyai?dependencies=0', timeout=10)
        if time.monotonic() - last_print > 15:
            print(json.dumps({'stage': label + '-runtime', 'state': r.get('state'),
                              'seconds': round(time.monotonic() - started)}), flush=True)
            last_print = time.monotonic()
        if r.get('state') == 'ready': break
        if r.get('state') in ('error', 'stopped'): raise RuntimeError(str(r.get('last_error')))
        time.sleep(3)
    else: raise RuntimeError('Runtime compilation/startup timeout')
    save(label + '-runtime', r)
    act('multistream.add', {'provider': 'joyai', 'options': {'seedPrompt': prompt}})
    started = time.monotonic(); last_print = 0
    while time.monotonic() - started < 600:
        listing, tile = current()
        st = (tile or {}).get('runtime', {}).get('stats', {})
        if time.monotonic() - last_print > 15:
            print(json.dumps({'stage': label + '-session', 'status': (tile or {}).get('status'),
                              'generated': st.get('generatedFrames'),
                              'seconds': round(time.monotonic() - started)}), flush=True)
            last_print = time.monotonic()
            if label.startswith('native'):
                for endpoint in ('debug', 'debug/stacks'):
                    try: save(label + '-' + endpoint.replace('/', '-') + '-' + str(round(time.monotonic() - started)),
                              adapter._request('GET', 'http://127.0.0.1:7873/' + endpoint, timeout=5))
                    except Exception: pass
        if (tile or {}).get('status') == 'error':
            save(label + '-failure', listing)
            raise RuntimeError(str(st.get('firstError') or st.get('lastRecoveryCause') or
                                   (tile or {}).get('ndi') or (tile or {}).get('error') or 'Client error'))
        if (tile or {}).get('status') == 'live' and st.get('generatedFrames', 0) > 16:
            got = [st.get(k) for k in ('nativeWidth', 'nativeHeight', 'nativeFps', 'rifeMultiplier')]
            expected = [config[k] for k in ('width', 'height', 'fps', 'rife_multiplier')]
            if got != expected: raise RuntimeError(f'Native preset mismatch: {got} != {expected}')
            # A user can edit the new tile while compilation is finishing.
            # Only restore the captured strength while its initial state is owned.
            current_strength = st.get('strength') or {}
            if st.get('prompt') == prompt and current_strength.get('revision', 0) == 0:
                act('model.primary.strength.set', {'provider': 'joyai', 'value': strength})
            save(label + '-live', listing)
            print(json.dumps({'stage': label + '-live', 'config': config}), flush=True)
            return
        time.sleep(2)
    raise RuntimeError('First-frame warmup timeout')

from easel_agent.ops_guard import hold_status
if hold_status().get('held'): raise RuntimeError('Another operations hold is active')
prior = json.loads((out / 'native-quality-before.json').read_text())
if rollback:
    listing = json.loads((out / 'trial-before-live.json').read_text())
    tile = next(s for s in listing['streams'] if s.get('type') == 'joyai')
else:
    listing, tile = current()
    if not tile: raise RuntimeError('Missing live JoyAI tile')
    save('trial-before-live', listing)
st = tile['runtime']['stats']
prompt = st['prompt']; strength = st['strength'].get('target', st['strength']['value'])
act('ops.hold', {'ttlMinutes': 40, 'reason': 'Compile and trial 1056x600 at 12 native / 24 output fps'})
try:
    stop()
    if rollback:
        launch(prior, prompt, strength, 'restored')
    else:
        try:
            launch({'width':1056,'height':600,'fps':12,'rife_multiplier':2}, prompt, strength, 'native600')
        except Exception as error:
            save('trial-failure', {'error': str(error)})
            print('Trial failed; live runtime and profile preserved: ' + str(error), flush=True)
            raise
finally:
    act('ops.resume')
