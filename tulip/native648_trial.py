"""Explicit higher-resolution trial. Checks report only; never roll back automatically."""
from pathlib import Path
import json,time,subprocess
root=Path(__file__).resolve().parents[1]
h=root/'tulip/native600_trial.py'
ns={'__file__':str(h),'__name__':'native648_bootstrap'}
exec(compile(h.read_text().split('from easel_agent.ops_guard import hold_status')[0],str(h),'exec'),ns)
from easel_agent.ops_guard import hold_status
if hold_status().get('held'):raise RuntimeError('Another operations hold is active')
out=root/'tulip/evidence'/('native648-'+time.strftime('%Y%m%d-%H%M%S'));out.mkdir();ns['out']=out
current,act,save,stop,launch=[ns[k] for k in ('current','act','save','stop','launch')]
listing,tile=current();save('before-live',listing)
if not tile or tile.get('status')!='live':raise RuntimeError('Expected a live baseline')
st=tile['runtime']['stats'];prompt=st['prompt'];strength=st['strength'].get('target',st['strength']['value'])
save('native-before',json.loads(ns['config_path'].read_text()))
cfg={'width':1152,'height':648,'fps':12,'rife_multiplier':2}
print(json.dumps({'trial':cfg,'automaticRollback':False,'evidence':str(out)}),flush=True)
act('ops.hold',{'ttlMinutes':40,'reason':'User requested 1152x648 trial; diagnostic checks cannot stop or revert rendering'})
try:
    stop()
    launch(cfg,prompt,strength,'native648')
    summaries={}
    for mode,seconds in [('observe',90),('stress',None)]:
        file=out/(mode+'.json')
        command=['node','scripts/smoke-tests/joyai-quality-live.cjs',mode,str(file)]
        if seconds:command.append(str(seconds))
        result=subprocess.run(command,cwd=root.parent/'etherea-ai',timeout=400,check=False)
        data=json.loads(file.read_text()) if file.exists() else {}
        summaries[mode]={'exitCode':result.returncode,'summary':data.get('summary')}
        save('checks',summaries)
        print(json.dumps({'check':mode,**summaries[mode],'automaticRollback':False}),flush=True)
    save('result',{'profile':cfg,'checks':summaries,'automaticRollback':False})
except Exception as error:
    save('failure',{'error':str(error),'automaticRollback':False})
    print('Trial issue; current runtime and profile preserved: '+str(error),flush=True)
    raise
finally:act('ops.resume')
