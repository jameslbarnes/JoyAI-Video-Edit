"""Clean runtime trial with watchdog diagnostics, acceptance checks and reporting only."""
from pathlib import Path
import json, time, subprocess
root=Path(__file__).resolve().parents[1]
helper=root/'tulip/native600_trial.py'
ns={'__file__':str(helper),'__name__':'guard_trial_bootstrap'}
exec(compile(helper.read_text().split('from easel_agent.ops_guard import hold_status')[0],str(helper),'exec'),ns)
from easel_agent.ops_guard import hold_status
if hold_status().get('held'): raise RuntimeError('Another operations hold is active')
out=ns['out']/('guarded-'+time.strftime('%H%M%S'));out.mkdir();ns['out']=out
act,save,current,stop,launch=[ns[k] for k in ('act','save','current','stop','launch')]
listing,tile=current()
if not tile or tile.get('status')!='live': raise RuntimeError('Expected live baseline')
prior=json.loads(ns['config_path'].read_text());save('native-before',prior);save('before-live',listing)
st=tile['runtime']['stats'];prompt=st['prompt'];strength=st['strength'].get('target',st['strength']['value'])
def observe(mode,seconds=None):
    file=out/(mode+'.json')
    command=['node','scripts/smoke-tests/joyai-quality-live.cjs',mode,str(file)]
    if seconds:command.append(str(seconds))
    subprocess.run(command,cwd=root.parent/'etherea-ai',check=True,timeout=360)
    data=json.loads(file.read_text());summary=data['summary']
    if summary.get('promptSuperseded'):raise RuntimeError('Competing prompt update invalidated comparison')
    if summary['errors'] or summary['outputDrops'] or summary['gpuDrops'] or summary['gpuUnderruns'] or summary['gaps']:
        raise RuntimeError(mode+' delivery acceptance failed')
    # Scene bridges keep the publisher moving but intentionally show no native
    # generated frames. Evaluate native throughput within unchanged epochs.
    runs=[]
    for sample in data['samples']:
        if sample['elapsed']<8:continue
        if not runs or runs[-1][-1]['epoch']!=sample['epoch']:runs.append([])
        runs[-1].append(sample)
    steady=[]
    for run in runs:
        a,b=run[0],run[-1];dt=(b['at']-a['at'])/1000
        if dt>=20:steady.append({'seconds':dt,'epoch':a['epoch'],
            'nativeFps':(b['model']-a['model'])/dt,'canvasFps':(b['display']-a['display'])/dt})
    save(mode+'-steady',{'runs':steady,'sceneChanges':len(runs)-1,'publisherFps':summary['gpuFps']})
    if not steady or any(s['nativeFps']<11.5 for s in steady) or summary['gpuFps']<23.5:
        raise RuntimeError(mode+' below 24fps acceptance')
act('ops.hold',{'ttlMinutes':40,'reason':'Qualify native 600p with pending-work watchdog, diagnostics and reporting only'})
try:
    try:
        stop()
        launch({'width':1056,'height':600,'fps':12,'rife_multiplier':2},prompt,strength,'native600')
        observe('observe',60)
        observe('stress')
        save('result',{'accepted':True,'config':json.loads(ns['config_path'].read_text())})
        print('600p accepted; steady and prompt stress checks passed',flush=True)
    except Exception as error:
        save('failure',{'error':str(error)})
        save('result',{'accepted':False,'automaticRollback':False,'error':str(error)})
        print('Trial check failed; live runtime and profile preserved: '+str(error),flush=True)
        raise
finally:act('ops.resume')
