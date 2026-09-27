"""Deploy native tap once and capture actual show frames; no preset changes."""
import hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
helper=ROOT/'tulip/native600_trial.py'
ns={'__file__':str(helper),'__name__':'native_capture_bootstrap'}
exec(compile(helper.read_text().split('from easel_agent.ops_guard import hold_status')[0],str(helper),'exec'),ns)
from easel_agent.ops_guard import hold_status
if hold_status().get('held'):raise RuntimeError('Another operations hold is active')
out=ROOT/'tulip/evidence/live-show-ab-20260924';out.mkdir(exist_ok=False);ns['out']=out
current,act,save,stop,launch=[ns[k] for k in ('current','act','save','stop','launch')]
cfg=json.loads(ns['config_path'].read_text(encoding='utf-8-sig'))
assert cfg=={'width':1152,'height':648,'fps':12,'rife_multiplier':2},'Preserve changed native settings'
settings=ROOT/'tulip/native-capture.json'
assert not settings.exists() or not json.loads(settings.read_text()),'Capture command already owned'
before={name:(ROOT/'tulip'/name).read_bytes() for name in ['native-quality.json','interpolation-settings.json','output-presentation.json','performance-settings.json']}
save('before-settings',{k:json.loads(v) for k,v in before.items()})
for endpoint in ('health','debug'):save('before-'+endpoint,ns['adapter']._request('GET','http://127.0.0.1:7873/'+endpoint))
save('code',{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['deploy/xvideo/serving/native_capture.py','deploy/xvideo/serving/interpolation_output.py']})
capture_id='live-show-ab-20260924'
command={'id':capture_id,'max_frames':192}
manifest=ROOT/'tulip/evidence/native-captures'/capture_id/'manifest.json'
act('ops.hold',{'ttlMinutes':20,'reason':'User requested comparison using the actual live show; add bounded native recording tap'})
try:
    listing,tile=current()
    if not tile or tile.get('status')!='live':raise RuntimeError('Expected a live JoyAI tile')
    save('before-live',listing);st=tile['runtime']['stats']
    prompt=st['prompt'];strength=st['strength'].get('target',st['strength']['value'])
    stop();launch(cfg,prompt,strength,'capture-tap')
    # Allow the startup/strength transition to finish before arming the tap.
    time.sleep(10)
    save('capture-live',current()[0])
    pending=settings.with_suffix('.pending.json');pending.write_text(json.dumps(command));pending.replace(settings)
    started=time.monotonic();last=0
    while not manifest.exists():
        if time.monotonic()-started>90:raise RuntimeError('Native capture did not finish; renderer remains live')
        if time.monotonic()-last>10:
            print(json.dumps({'stage':'capture','seconds':round(time.monotonic()-started)}),flush=True);last=time.monotonic()
        time.sleep(1)
    data=json.loads(manifest.read_text());save('capture-manifest',data)
    assert data['complete'] and data['written']==192 and data['error'] is None,data
    for endpoint in ('health','debug'):save('after-'+endpoint,ns['adapter']._request('GET','http://127.0.0.1:7873/'+endpoint))
    save('after-live',current()[0])
    unchanged={k:json.loads((ROOT/'tulip'/k).read_bytes())==json.loads(v) for k,v in before.items()}
    save('settings-preserved',unchanged)
    print(json.dumps({'stage':'captured','frames':data['written'],'capture':str(manifest),'settingsPreserved':unchanged}),flush=True)
finally:
    if settings.exists() and json.loads(settings.read_text())==command:
        pending=settings.with_suffix('.pending.json');pending.write_text('{}\n');pending.replace(settings)
    act('ops.resume')
