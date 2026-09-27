"""Explicit 720p/VSR trial (or explicit 648p return); never automatic rollback."""
from pathlib import Path
import hashlib
import json
import sys
import time

root = Path(__file__).resolve().parents[1]
helper = root/'tulip/native600_trial.py'
ns = {'__file__':str(helper),'__name__':'native720_vsr_bootstrap'}
exec(compile(helper.read_text().split('from easel_agent.ops_guard import hold_status')[0],str(helper),'exec'),ns)
from easel_agent.ops_guard import hold_status
if hold_status().get('held'): raise RuntimeError('Another operations hold is active')
restore = '--restore-648' in sys.argv
compact = '--compact-prompt-graphs' in sys.argv
if compact and not restore:
    raise RuntimeError('--compact-prompt-graphs requires --restore-648')
out = root/'tulip/evidence'/('native648-vsr-return-' if restore else 'native720-vsr-')
out = out.with_name(out.name+time.strftime('%Y%m%d-%H%M%S'))
out.mkdir(parents=True,exist_ok=False)
ns['out'] = out
current,act,save,stop,launch = [ns[k] for k in ('current','act','save','stop','launch')]
listing,tile = current()
if not tile: raise RuntimeError('Current JoyAI tile missing; preserve scene until explicitly recovered')
save('before-live',listing)
stats=tile['runtime']['stats']
prompt=stats['prompt']
strength=stats['strength'].get('target',stats['strength']['value'])
save('native-before',json.loads(ns['config_path'].read_text()))
presentation=json.loads((root/'tulip/output-presentation.json').read_text(encoding='utf-8-sig'))
if presentation.get('backend')!='vsr_ultra': raise RuntimeError('SR selection changed; preserve it')
cfg=dict(width=1152 if restore else 1296,height=648 if restore else 720,fps=12,rife_multiplier=2)
performance_path=root/'tulip/performance-settings.json'
restored_performance=None
if restore:
    # Return the complete tuning profile while retaining later lifecycle fixes.
    favorite=root/'tulip/favorites/648p-vsr-20260922'
    manifest=json.loads((favorite/'manifest.json').read_text(encoding='utf-8'))
    entry=next(e for e in manifest['files'] if e['path']=='tulip/performance-settings.json')
    restored_performance=(favorite/'files'/entry['path']).read_bytes()
    if hashlib.sha256(restored_performance).hexdigest()!=entry['sha256']:
        raise RuntimeError('Saved 648p performance settings failed checksum verification')
    if compact:
        # Keep the fast 512-token path and full 1024-token support, with fewer
        # private graph pools so scene replacement has VAE scratch headroom.
        tuned=json.loads(restored_performance)
        tuned.update(compactPromptGraphs=True,singlePromptGraph=False)
        restored_performance=json.dumps(tuned,indent=2).encode('utf-8')
    save('performance-before',json.loads(performance_path.read_text(encoding='utf-8-sig')))
    save('performance-restored',json.loads(restored_performance))
print(json.dumps(dict(profile=cfg,evidence=str(out),compactPromptGraphs=compact,automaticRollback=False)),flush=True)
act('ops.hold',dict(ttlMinutes=30,reason='User-requested 720p/VSR headroom trial' if not restore else 'Explicit return to qualified 648p/VSR profile'))
try:
    stop()
    if restored_performance is not None:
        pending=performance_path.with_suffix('.restore-pending.json')
        pending.write_bytes(restored_performance);pending.replace(performance_path)
    launch(cfg,prompt,strength,'native648-return' if restore else 'native720-vsr')
    save('after-health',ns['adapter']._request('GET','http://127.0.0.1:7873/health'))
finally:
    act('ops.resume')
