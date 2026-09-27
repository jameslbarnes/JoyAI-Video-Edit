"""Explicit repair of failed 648p runtime; preserve resolution and creative settings."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[1];h=root/'tulip/native600_trial.py'
ns={'__file__':str(h),'__name__':'repair648_bootstrap'}
exec(compile(h.read_text().split('from easel_agent.ops_guard import hold_status')[0],str(h),'exec'),ns)
from easel_agent.ops_guard import hold_status
if hold_status().get('held'):raise RuntimeError('Another operations hold is active')
out=root/'tulip/evidence/native648-20260922-202107/restart';out.mkdir(exist_ok=True);ns['out']=out
current,act,save,stop,launch=[ns[k] for k in ('current','act','save','stop','launch')]
listing,tile=current();save('before-live',listing)
if not tile:raise RuntimeError('Missing failed JoyAI tile')
s=tile['runtime']['stats'];prompt=s['prompt']
b=next(t for t in json.loads((out.parent/'before-live.json').read_text())['streams'] if t['type']=='joyai')['runtime']['stats']
if any(prompt=='\n'.join([b['prompt']]*n) for n in (1,2,3,4,5,6,8)):prompt=b['prompt']
strength=(s.get('strength') or b['strength']);strength=strength.get('target',strength['value'])
cfg=json.loads(ns['config_path'].read_text())
if (cfg['width'],cfg['height'])!=(1152,648):raise RuntimeError('Profile changed; preserving it')
act('ops.hold',{'ttlMinutes':20,'reason':'Clear captured GPU OOM by restarting the same 648p profile; no resolution rollback'})
try:
    stop();launch(cfg,prompt,strength,'native648-repaired')
finally:act('ops.resume')
