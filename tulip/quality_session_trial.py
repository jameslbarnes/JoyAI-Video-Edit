"""Change a validated native preset at a clean session boundary, retaining weights."""
from pathlib import Path
import json
import sys
import time
root=Path(__file__).resolve().parents[1]
helper=root/'tulip/quality_trial_lifecycle.py'
ns={'__file__':str(helper),'__name__':'quality_helper'}
exec(compile(helper.read_text().split('listing =')[0],str(helper),'exec'),ns)
out=ns['out']/('session-'+time.strftime('%H%M%S'));out.mkdir();ns['out']=out
adapter,act,save=ns['adapter'],ns['act'],ns['save']
width,height,fps,rife=map(int,sys.argv[1:5])
config={'width':width,'height':height,'fps':fps,'rife_multiplier':rife}
sys.path.insert(0,str(root/'deploy'))
from xvideo.serving.native_config import load_native_config
pending=root/'tulip/native-quality.pending.json';pending.write_text(json.dumps(config))
load_native_config(pending)
listing=adapter._request('GET',ns['control']+'/control/multistream/list');save('before-live',listing)
tile=next(s for s in listing['streams'] if s['type']=='joyai');stats=tile['runtime']['stats']
if len(sys.argv)>5:
    prior=json.loads(Path(sys.argv[5]).read_text(encoding='utf-8'))
    stats=next(s for s in prior['streams'] if s['type']=='joyai')['runtime']['stats']
prompt=stats['prompt'];strength=stats['strength'].get('target',stats['strength']['value'])
save('native-before',json.loads((root/'tulip/native-quality.json').read_text()))
act('ops.hold',{'ttlMinutes':10,'reason':'Compare native quality preset without reloading model weights'})
try:
    act('multistream.remove',{'provider':'joyai','force':True})
    pending.replace(root/'tulip/native-quality.json')
    act('multistream.add',{'provider':'joyai','options':{'seedPrompt':prompt}})
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        live=adapter._request('GET',ns['control']+'/control/multistream/list')
        tile=next((s for s in live['streams'] if s['type']=='joyai'),{})
        st=tile.get('runtime',{}).get('stats',{})
        if tile.get('status')=='live' and st.get('generatedFrames',0)>16:
            if [st.get('nativeWidth'),st.get('nativeHeight'),st.get('nativeFps'),st.get('rifeMultiplier')]!=[width,height,fps,rife]:
                raise RuntimeError('Service did not apply requested native preset')
            break
        if tile.get('status')=='error': raise RuntimeError(str(tile))
        time.sleep(2)
    else: raise RuntimeError('Session initialization timeout')
    act('model.primary.strength.set',{'provider':'joyai','value':strength})
    save('after-live',live);print(json.dumps({'live':config,'evidence':str(out)}),flush=True)
finally:
    act('ops.resume')
