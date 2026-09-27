"""One explicit live DLSSG deployment, preserving current creative controls."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
helper=root/'tulip/native600_trial.py'
ns={'__file__':str(helper),'__name__':'dlssg_live_bootstrap'}
exec(compile(helper.read_text().split('from easel_agent.ops_guard import hold_status')[0],str(helper),'exec'),ns)
from easel_agent.ops_guard import hold_status
if hold_status().get('held'):raise RuntimeError('Another operations hold is active')
out=root/'tulip/evidence/dlssg-live-20260924/deploy';out.mkdir(exist_ok=False);ns['out']=out
current,act,save,stop,launch=[ns[k] for k in ('current','act','save','stop','launch')]
cfg=json.loads(ns['config_path'].read_text(encoding='utf-8-sig'))
if cfg!={'width':1152,'height':648,'fps':12,'rife_multiplier':2}:raise RuntimeError('Native profile changed; preserve it')
save('before-native',cfg)
for endpoint in ('health','debug'):
    save('before-'+endpoint,ns['adapter']._request('GET','http://127.0.0.1:7873/'+endpoint))
act('ops.hold',{'ttlMinutes':30,'reason':'User explicitly requested live GPU DLSSG visual comparison; stalls accepted'})
try:
    listing,tile=current()
    if not tile or tile.get('status')!='live':raise RuntimeError('Expected current live JoyAI tile')
    save('before-live',listing);st=tile['runtime']['stats']
    prompt=st['prompt'];strength=st['strength'].get('target',st['strength']['value'])
    stop()
    path=root/'tulip/interpolation-settings.json';pending=path.with_suffix('.pending.json')
    pending.write_text('{"backend":"dlssg"}\n');pending.replace(path)
    launch(cfg,prompt,strength,'dlssg-live')
    for endpoint in ('health','debug'):
        save('after-'+endpoint,ns['adapter']._request('GET','http://127.0.0.1:7873/'+endpoint))
finally:
    act('ops.resume')
