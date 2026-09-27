"""Explicit same-profile 648p session replacement, preserving current scene controls."""
import argparse
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('evidence',type=Path)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
helper=root/'tulip/native600_trial.py'
ns={'__file__':str(helper),'__name__':'verify648_bootstrap'}
exec(compile(helper.read_text().split('from easel_agent.ops_guard import hold_status')[0],str(helper),'exec'),ns)
from easel_agent.ops_guard import hold_status
if hold_status().get('held'): raise RuntimeError('Another operations hold is active')
out=args.evidence.resolve()/'replacement'
out.mkdir(parents=True,exist_ok=False)
ns['out']=out
current,act,save,launch=[ns[k] for k in ('current','act','save','launch')]
listing,tile=current()
if not tile or tile.get('status')!='live': raise RuntimeError('Expected a live JoyAI tile')
save('before-live',listing)
stats=tile['runtime']['stats']
prompt=stats['prompt'];strength=stats['strength'].get('target',stats['strength']['value'])
cfg=json.loads(ns['config_path'].read_text(encoding='utf-8-sig'))
if cfg!={'width':1152,'height':648,'fps':12,'rife_multiplier':2}:
    raise RuntimeError('648p profile changed; preserve it')
save('before-health',ns['adapter']._request('GET','http://127.0.0.1:7873/health'))
act('ops.hold',dict(ttlMinutes=15,reason='Verify 648p session replacement with compact prompt graphs'))
try:
    act('multistream.remove',{'provider':'joyai','force':True})
    launch(cfg,prompt,strength,'native648-replacement')
    save('after-health',ns['adapter']._request('GET','http://127.0.0.1:7873/health'))
    save('after-debug',ns['adapter']._request('GET','http://127.0.0.1:7873/debug'))
finally:
    act('ops.resume')
