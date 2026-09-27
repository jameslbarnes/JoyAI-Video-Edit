"""Resume the same 648p profile after captured OOM; no runtime restart or rollback."""
from pathlib import Path
import json,time
root=Path(__file__).resolve().parents[1];h=root/'tulip/native600_trial.py'
ns={'__file__':str(h),'__name__':'resume648_bootstrap'}
exec(compile(h.read_text().split('from easel_agent.ops_guard import hold_status')[0],str(h),'exec'),ns)
from easel_agent.ops_guard import hold_status
if hold_status().get('held'):raise RuntimeError('Another operations hold is active')
out=root/'tulip/evidence/native648-20260922-202107/resume';out.mkdir(exist_ok=True);ns['out']=out
current,act,save=[ns[k] for k in ('current','act','save')]
listing,tile=current();save('before-live',listing)
if not tile:raise RuntimeError('Missing JoyAI tile')
s=tile['runtime']['stats'];prompt=s['prompt']
base=next(s for s in json.loads((out.parent/'before-live.json').read_text())['streams'] if s['type']=='joyai')['runtime']['stats']['prompt']
if any(prompt=='\n'.join([base]*n) for n in (1,2,3,4,5,6,8)):prompt=base
strength=s['strength'].get('target',s['strength']['value'])
cfg=json.loads(ns['config_path'].read_text())
if (cfg['width'],cfg['height'])!=(1152,648):raise RuntimeError('Profile changed; preserving it')
act('ops.hold',{'ttlMinutes':10,'reason':'Resume same 648p session after captured GPU memory failure; no downgrade or runtime restart'})
try:
    act('multistream.remove',{'provider':'joyai','force':True})
    act('multistream.add',{'provider':'joyai','options':{'seedPrompt':prompt}})
    for i in range(120):
        listing,tile=current();s=(tile or {}).get('runtime',{}).get('stats',{});save('latest',listing)
        if i%5==0:print(json.dumps({'status':(tile or {}).get('status'),'frames':s.get('generatedFrames'),'error':s.get('firstError')}),flush=True)
        if (tile or {}).get('status')=='error':raise RuntimeError(str(s.get('firstError') or (tile or {}).get('ndi')))
        if (tile or {}).get('status')=='live' and s.get('generatedFrames',0)>24:
            act('model.primary.strength.set',{'provider':'joyai','value':strength})
            save('live',listing);print('648p rendering resumed; no resolution rollback',flush=True);break
        time.sleep(2)
    else:raise RuntimeError('Session initialization timeout; no automatic profile change')
finally:act('ops.resume')
