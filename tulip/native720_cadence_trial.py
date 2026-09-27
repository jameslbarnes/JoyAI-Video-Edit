"""Explicit cadence comparison at unchanged 720p, reusing loaded model weights."""
import argparse
import json
from pathlib import Path
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--fps',type=int,choices=(6,8,10,12),required=True)
parser.add_argument('--rife',type=int,choices=(1,2,4),required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
helper=root/'tulip/native600_trial.py'
ns={'__file__':str(helper),'__name__':'native720_cadence_bootstrap'}
exec(compile(helper.read_text().split('from easel_agent.ops_guard import hold_status')[0],str(helper),'exec'),ns)
from easel_agent.ops_guard import hold_status
if hold_status().get('held'): raise RuntimeError('Another operations hold is active')
out=root/'tulip/evidence'/('native720-cadence-'+str(args.fps)+'x'+str(args.rife)+'-'+time.strftime('%Y%m%d-%H%M%S'))
out.mkdir();ns['out']=out
current,act,save,launch=[ns[k] for k in ('current','act','save','launch')]
listing,tile=current();save('before-live',listing)
if not tile or tile.get('status')!='live': raise RuntimeError('Expected a live JoyAI tile')
stats=tile['runtime']['stats']
prompt=stats['prompt'];strength=stats['strength'].get('target',stats['strength']['value'])
cfg=json.loads(ns['config_path'].read_text(encoding='utf-8-sig'))
if (cfg['width'],cfg['height'])!=(1296,720): raise RuntimeError('Native resolution changed')
save('before-native',cfg)
cfg={**cfg,'fps':args.fps,'rife_multiplier':args.rife}
print(json.dumps(dict(evidence=str(out),profile=cfg,automaticRollback=False,reuseLoadedModel=True)),flush=True)
act('ops.hold',dict(ttlMinutes=20,reason='Explicit 720p cadence comparison: '+str(args.fps)+' native fps, RIFE '+str(args.rife)))
try:
    act('multistream.remove',{'provider':'joyai','force':True})
    launch(cfg,prompt,strength,'native720-cadence')
    save('after-health',ns['adapter']._request('GET','http://127.0.0.1:7873/health'))
finally:
    act('ops.resume')
