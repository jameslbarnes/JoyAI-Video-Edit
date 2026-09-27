"""Load the reference-control experiment through the existing SDK lifecycle.

Preserves native/output settings and prompt. No timeout-triggered rollback.
The initial reference-only value of 1 matches the previous zero-blend baseline.
"""
import json
import os
from pathlib import Path
import sys
import time
import subprocess

root = Path(__file__).resolve().parents[1]
sdk = root.parent / 'easel-agent-sdk'
combined = '--combined' in sys.argv
out = Path((root / ('tulip/combined-strength-current.txt' if combined else 'tulip/reference-strength-current.txt')).read_text())
reload_only = '--reload' in sys.argv
if reload_only:
    out = out / time.strftime('reload-%H%M%S')
    out.mkdir()
sys.path.insert(0, str(sdk / 'src'))
for line in (sdk / '.env').read_text(encoding='utf-8-sig').splitlines():
    line = line.strip()
    if line and not line.startswith('#') and '=' in line:
        key, value = line.removeprefix('export ').split('=', 1)
        os.environ[key.strip()] = value.strip().strip(chr(39)+chr(34))
from easel_agent.cli import _router
from easel_agent.primary_models import LivePrimaryAdapter
from easel_agent.ops_guard import hold_status
router = _router(dry_run=False, config_path=str(sdk / 'nanos.json'))
adapter = LivePrimaryAdapter(router.store)
backend, control = 'http://127.0.0.1:7860', 'http://127.0.0.1:8793'

def save(name, value):
    (out / (name+'.json')).write_text(json.dumps(value, indent=2), encoding='utf-8')

def act(name, payload=None):
    result = router.dispatch(name, {'config': str(sdk/'nanos.json'),
        'source': 'user-requested-reference-strength', **(payload or {})})
    save(name, {**result, 'data':{k:v for k,v in result.get('data',{}).items() if k != 'state'}})
    if not result.get('ok'): raise RuntimeError(name + ': ' + str(result.get('error')))
    print(json.dumps({'action': name, 'ok': True}), flush=True)
    return result

def current():
    listing = adapter._request('GET', control+'/control/multistream/list')
    return listing, next((s for s in listing['streams'] if s.get('type') == 'joyai'), None)

if hold_status().get('held'): raise RuntimeError('Another operations hold is active')
listing, tile = current()
if not tile or tile.get('status') != 'live': raise RuntimeError('Expected live JoyAI')
stats = tile['runtime']['stats']
if stats.get('referenceImageActive'): raise RuntimeError('Reference image requires separate preservation')
if combined and stats['strength']['mode'] != 'reference_only':
    raise RuntimeError('Combined migration expects reference-only input units')
if not combined and not reload_only and (stats['strength']['target'] != 0 or stats['strength']['mode'] != 'direct_latent_blend'):
    raise RuntimeError('Baseline changed; re-evaluate migration before touching the show')
save('restart-before', listing)
prompt = stats['prompt']
files = ['native-quality.json','performance-settings.json','output-presentation.json',
         'interpolation-settings.json','neural-rendering.json']
original = {name:(root/'tulip'/name).read_bytes() for name in files}
config = root/'tulip/shader-strength.json'
if config.exists() and not reload_only and not combined: raise RuntimeError('Strength config already exists; do not repeat deployment')
startup = ({'mode':'combined','value':stats['strength']['target']*.6} if combined else
           {'mode':stats['strength']['mode'], 'value':stats['strength']['target']} if reload_only
           else {'mode':'reference_only','value':1})
act('ops.hold', {'ttlMinutes':20, 'reason':'Load and test shader reference-strength control'})
try:
    pending = config.with_suffix('.pending.json')
    pending.write_text(json.dumps(startup, indent=2)+'\n', encoding='utf-8')
    pending.replace(config)
    act('multistream.remove', {'provider':'joyai','force':True})
    result = adapter._request('POST', backend+'/api/runtimes/joyai/stop',
        {'force':True,'cleanupOrphans':False}, timeout=40)
    save('runtime-stop',result)
    if result.get('process_running'): raise RuntimeError('JoyAI did not stop')
    if combined:
        proof = subprocess.run([str(root/'.venv/Scripts/python.exe'), str(root/'tulip/test_shader_presentation_gpu.py')],
            cwd=root, capture_output=True, text=True, timeout=60)
        save('gpu-proof',{'exit_code':proof.returncode,'output':proof.stdout,'error':proof.stderr})
        if proof.returncode: raise RuntimeError('GPU endpoint proof failed: '+proof.stderr[-1000:])
        print(proof.stdout.strip(),flush=True)
    adapter._request('POST', backend+'/api/runtimes/joyai/start', {'wait':False}, timeout=20)
    started = time.monotonic()
    while time.monotonic()-started < 900:
        result = adapter._request('GET', backend+'/api/runtimes/joyai?dependencies=0',timeout=10)
        print(json.dumps({'stage':'runtime','state':result.get('state'),'elapsed':round(time.monotonic()-started)}),flush=True)
        if result.get('state') == 'ready': break
        if result.get('state') in ('error','stopped'): raise RuntimeError(str(result.get('last_error')))
        time.sleep(10)
    else: raise RuntimeError('Startup wait expired; runtime left untouched, no rollback')
    save('runtime-start',result)
    act('multistream.add', {'provider':'joyai','options':{'seedPrompt':prompt}})
    started=time.monotonic()
    while time.monotonic()-started < 300:
        listing,tile=current(); stats=(tile or {}).get('runtime',{}).get('stats',{})
        if tile and tile.get('status')=='live' and stats.get('generatedFrames',0)>32: break
        if tile and tile.get('status')=='error': raise RuntimeError(str(stats.get('firstError')))
        print(json.dumps({'stage':'session','status':(tile or {}).get('status'),'frames':stats.get('generatedFrames')}),flush=True)
        time.sleep(5)
    else: raise RuntimeError('First-frame wait expired; runtime left untouched, no rollback')
    assert stats['strength']['mode']==startup['mode']
    if stats['strength']['revision']==0:
        assert stats['strength']['blend']==0
    assert all((root/'tulip'/name).read_bytes()==value for name,value in original.items())
    save('restart-after',listing)
    print(json.dumps({'stage':'live','mode':stats['strength']['mode'],'strength':stats['strength']['target'],
                      'native':[stats['nativeWidth'],stats['nativeHeight']], 'fps':stats['outputFps'],
                      'prompt_preserved':stats['prompt']==prompt}),flush=True)
finally:
    act('ops.resume')
