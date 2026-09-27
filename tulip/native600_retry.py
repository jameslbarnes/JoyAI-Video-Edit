"""Retry the cached 600p preset at a session boundary; retain exact failures without automatic rollback."""
from pathlib import Path
import json, time
root = Path(__file__).resolve().parents[1]
helper = root / 'tulip/native600_trial.py'
ns = {'__file__':str(helper),'__name__':'native600_retry_bootstrap'}
exec(compile(helper.read_text().split('from easel_agent.ops_guard import hold_status')[0],str(helper),'exec'), ns)
from easel_agent.ops_guard import hold_status
if hold_status().get('held'): raise RuntimeError('Another operations hold is active')
current,act,adapter = ns['current'],ns['act'],ns['adapter']
out = ns['out'] / ('retry-' + time.strftime('%H%M%S')); out.mkdir(); ns['out'] = out
save = ns['save']
listing,tile = current(); save('before-live',listing)
if not tile or tile.get('status') != 'live': raise RuntimeError('Expected live baseline')
prior = json.loads(ns['config_path'].read_text()); save('native-before',prior)
st = tile['runtime']['stats']; prompt = st['prompt']; strength = st['strength'].get('target',st['strength']['value'])
def switch(cfg,label):
    if current()[1]: act('multistream.remove',{'provider':'joyai','force':True})
    pending = ns['config_path'].with_suffix('.retry-pending.json')
    pending.write_text(json.dumps(cfg)); pending.replace(ns['config_path'])
    act('multistream.add',{'provider':'joyai','options':{'seedPrompt':prompt}})
    start=time.monotonic(); last=0
    while time.monotonic()-start < 600:
        listing,tile=current(); st=(tile or {}).get('runtime',{}).get('stats',{})
        save(label+'-latest',listing)
        if time.monotonic()-last>15:
            print(json.dumps({'stage':label,'seconds':round(time.monotonic()-start),'status':(tile or {}).get('status'),'generated':st.get('generatedFrames'),'received':st.get('receivedFrames'),'firstError':st.get('firstError')}),flush=True); last=time.monotonic()
        if (tile or {}).get('status')=='error':
            save(label+'-failure',listing)
            for endpoint in ('debug','debug/stacks','health'):
                try: save(label+'-'+endpoint.replace('/','-'),adapter._request('GET','http://127.0.0.1:7873/'+endpoint,timeout=5))
                except Exception: pass
            raise RuntimeError(json.dumps({'firstError':st.get('firstError'),'detail':(tile or {}).get('ndi'),'lastRecoveryCause':st.get('lastRecoveryCause'),'error':(tile or {}).get('error')}))
        if (tile or {}).get('status')=='live' and st.get('generatedFrames',0)>48:
            if [st.get(k) for k in ('nativeWidth','nativeHeight','nativeFps','rifeMultiplier')] != [cfg[k] for k in ('width','height','fps','rife_multiplier')]: raise RuntimeError('Native preset mismatch')
            act('model.primary.strength.set',{'provider':'joyai','value':strength})
            save(label+'-live',listing); print(json.dumps({'live':cfg,'evidence':str(out)}),flush=True); return
        time.sleep(2)
    raise RuntimeError('Session initialization timeout')
act('ops.hold',{'ttlMinutes':25,'reason':'Retry cached native 600p session with diagnostics and no automatic rollback'})
try:
    try: switch({'width':1056,'height':600,'fps':12,'rife_multiplier':2},'native600')
    except Exception as error:
        save('failure',{'error':str(error)})
        print('Trial failed; live runtime and profile preserved: '+str(error),flush=True)
        raise
finally: act('ops.resume')
