"""Explicit 648p padded-text trial, using the existing runtime; no automatic rollback."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('evidence',type=Path)
p.add_argument('--enabled',choices=('yes','no'),required=True)
args=p.parse_args()
root=Path(__file__).resolve().parents[1]
helper=root/'tulip/native600_trial.py'
ns={'__file__':str(helper),'__name__':'fp8_prefix_bootstrap'}
exec(compile(helper.read_text().split('from easel_agent.ops_guard import hold_status')[0],str(helper),'exec'),ns)
from easel_agent.ops_guard import hold_status
if hold_status().get('held'):raise RuntimeError('Another operations hold is active')
out=args.evidence.resolve();out.mkdir(parents=True,exist_ok=False);ns['out']=out
current,act,save,stop,launch=[ns[k] for k in ('current','act','save','stop','launch')]
listing,tile=current()
if not tile or tile.get('status')!='live':raise RuntimeError('Expected the existing live JoyAI tile')
cfg=json.loads(ns['config_path'].read_text(encoding='utf-8-sig'))
if cfg!={'width':1152,'height':648,'fps':12,'rife_multiplier':2}:raise RuntimeError('Native profile changed; preserve it')
path=root/'tulip/performance-settings.json'
performance=json.loads(path.read_text(encoding='utf-8-sig'))
save('before-performance',performance);save('before-native',cfg)
performance['trimPaddedText']=args.enabled=='yes'
for endpoint in ('health','debug'):
    save('before-'+endpoint,ns['adapter']._request('GET','http://127.0.0.1:7873/'+endpoint))
act('ops.hold',{'ttlMinutes':30,'reason':'User-approved same-quality 648p padded-text optimization trial'})
try:
    # Capture the current creative controls immediately before replacing the tile.
    listing,tile=current();save('before-live',listing)
    if not tile or tile.get('status')!='live':raise RuntimeError('Live tile changed')
    st=tile['runtime']['stats'];prompt=st['prompt'];strength=st['strength'].get('target',st['strength']['value'])
    stop()
    pending=path.with_suffix('.pending.json');pending.write_text(json.dumps(performance,indent=2)+'\n');pending.replace(path)
    launch(cfg,prompt,strength,'fp8-prefix-'+args.enabled)
    for endpoint in ('health','debug'):
        save('after-'+endpoint,ns['adapter']._request('GET','http://127.0.0.1:7873/'+endpoint))
finally:
    act('ops.resume')
