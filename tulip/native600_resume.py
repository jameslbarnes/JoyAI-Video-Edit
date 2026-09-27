"""Resume the explicitly requested 600p session without restarting model weights."""
from pathlib import Path
import json,time
root=Path(__file__).resolve().parents[1]
h=root/'tulip/native600_trial.py'
ns={'__file__':str(h),'__name__':'resume600_bootstrap'}
exec(compile(h.read_text().split('from easel_agent.ops_guard import hold_status')[0],str(h),'exec'),ns)
from easel_agent.ops_guard import hold_status
out=root/'tulip/evidence/native600-20260922/user-resume600';out.mkdir(exist_ok=True);ns['out']=out
current,act,save=[ns[k] for k in ('current','act','save')]
listing,tile=current();save('before-live',listing)
if not tile:raise RuntimeError('Missing current JoyAI session')
st=tile['runtime']['stats'];prompt=st['prompt']
before=root/'tulip/evidence/native600-20260922/guarded-200649/failure-live.json'
b=json.loads(before.read_text()) if before.exists() else listing
bs=next(t for t in b['streams'] if t['type']=='joyai')['runtime']['stats']
strength=(bs.get('strength') or st['strength']);strength=strength.get('target',strength['value'])
# The interrupted trial's hold belongs to this task; release it before replacing it.
if hold_status().get('held'):act('ops.resume')
act('ops.hold',{'ttlMinutes':15,'reason':'User requested restoring 600p; automatic rollback removed'})
try:
    act('multistream.remove',{'provider':'joyai','force':True})
    cfg={'width':1056,'height':600,'fps':12,'rife_multiplier':2}
    pending=ns['config_path'].with_suffix('.resume-pending.json');pending.write_text(json.dumps(cfg));pending.replace(ns['config_path'])
    act('multistream.add',{'provider':'joyai','options':{'seedPrompt':prompt}})
    for i in range(180):
        listing,tile=current();save('latest',listing);s=(tile or {}).get('runtime',{}).get('stats',{})
        if i%5==0:print(json.dumps({'status':(tile or {}).get('status'),'native':[s.get('nativeWidth'),s.get('nativeHeight')],'frames':s.get('generatedFrames'),'error':s.get('firstError')}),flush=True)
        if (tile or {}).get('status')=='error':raise RuntimeError(str(s.get('firstError') or (tile or {}).get('ndi')))
        if (tile or {}).get('status')=='live' and s.get('generatedFrames',0)>24:
            act('model.primary.strength.set',{'provider':'joyai','value':strength})
            save('live',listing);print('600p restored; automatic rollback disabled',flush=True);break
        time.sleep(2)
    else:raise RuntimeError('Session still initializing; no automatic rollback')
finally:act('ops.resume')
