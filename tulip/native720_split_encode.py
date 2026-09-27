"""Compare explicit 720p input VAE placement with compact prompt graphs."""
from pathlib import Path
import argparse
import json
import time
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--encode-device',choices=('model','output'),default='output')
parser.add_argument('--pseudo-device',choices=('inherit','model','output'),default='inherit')
parser.add_argument('--fps',type=int,choices=(6,8,10,12))
parser.add_argument('--rife',type=int,choices=(1,2,4))
parser.add_argument('--single-prompt-graph',action='store_true')
args=parser.parse_args()
if (args.fps is None)!=(args.rife is None): parser.error('--fps and --rife must be supplied together')
root=Path(__file__).resolve().parents[1]
helper=root/'tulip/native600_trial.py'
ns={'__file__':str(helper),'__name__':'native720_split_bootstrap'}
exec(compile(helper.read_text().split('from easel_agent.ops_guard import hold_status')[0],str(helper),'exec'),ns)
from easel_agent.ops_guard import hold_status
if hold_status().get('held'): raise RuntimeError('Another operations hold is active')
out=root/'tulip/evidence'/('native720-compact-'+args.encode_device+'-pseudo-'+args.pseudo_device+'-'+time.strftime('%Y%m%d-%H%M%S'))
out.mkdir();ns['out']=out
current,act,save,stop,launch=[ns[k] for k in ('current','act','save','stop','launch')]
listing,tile=current();save('before-live',listing)
if not tile: raise RuntimeError('Missing JoyAI tile')
stats=tile['runtime']['stats']
prompt=stats['prompt'];strength=stats['strength'].get('target',stats['strength']['value'])
prior=json.loads((root/'tulip/evidence/native720-vsr-20260922-232318/before-live.json').read_text())
baseline=next(t for t in prior['streams'] if t['type']=='joyai')['runtime']['stats']
if prompt==baseline['prompt'] and stats['strength'].get('revision',0)==0:
    strength=baseline['strength']['target']
cfg=json.loads(ns['config_path'].read_text())
if (cfg.get('width'),cfg.get('height'))!=(1296,720): raise RuntimeError('Native resolution changed')
save('before-native',cfg)
if args.fps is not None: cfg={**cfg,'fps':args.fps,'rife_multiplier':args.rife}
path=root/'tulip/performance-settings.json'
performance=json.loads(path.read_text(encoding='utf-8-sig'));save('before-performance',performance)
print(json.dumps(dict(evidence=str(out),profile=cfg,vaeEncodeDevice=args.encode_device,vaePseudoDevice=args.pseudo_device,compactPromptGraphs=True,singlePromptGraph=args.single_prompt_graph,automaticRollback=False)),flush=True)
act('ops.hold',dict(ttlMinutes=30,reason='Keep 720p and compact graphs; encoder='+args.encode_device+', pseudo encoder='+args.pseudo_device))
try:
    stop()
    performance['vaeEncodeDevice']=args.encode_device
    performance['vaePseudoDevice']=args.pseudo_device
    performance['compactPromptGraphs']=True
    performance['singlePromptGraph']=args.single_prompt_graph
    pending=path.with_suffix('.pending.json');pending.write_text(json.dumps(performance,indent=2));pending.replace(path)
    launch(cfg,prompt,strength,'native720-'+args.encode_device+'-encode')
    save('after-health',ns['adapter']._request('GET','http://127.0.0.1:7873/health'))
finally: act('ops.resume')
