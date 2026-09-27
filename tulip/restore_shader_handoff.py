"""Restore only the rejected 2026-09-25 motion-handoff changes, with hash guards."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

root = Path(__file__).resolve().parents[1]
etherea = root.parent / 'etherea-ai'
source = root / 'tulip/evidence/generated-handoff-20260925-114939'
out = root / 'tulip/evidence' / ('restore-shader-handoff-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
hashes = json.loads((source / 'deployed-source-hashes.json').read_text())
server_files = ['direct_ndi_output.py', 'gpu_video_output.py', 'upscale_output.py', 'serve_joyomni_streaming.py']
pairs = [(root / 'deploy/xvideo/serving' / name, source / 'before/deploy/xvideo/serving' / name) for name in server_files]
client = etherea / 'src/web-client/static/js/joyai-client.js'
manager = etherea / 'src/web-client/static/js/multistream-manager.js'
pairs.append((client, source / 'joyai-client.before.js'))
for target in [p[0] for p in pairs] + [manager]:
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    if actual != hashes[str(target)]:
        raise RuntimeError(f'Concurrent change; refusing to overwrite {target}')
out.mkdir()
for target, backup in pairs:
    (out / (target.name + '.rejected')).write_bytes(target.read_bytes())
    target.write_bytes(backup.read_bytes())
(out / 'multistream-manager.js.rejected').write_bytes(manager.read_bytes())
data = manager.read_bytes()
old = b'/static/js/joyai-client.js?v=generated-motion-1'
assert data.count(old) == 1
manager.write_bytes(data.replace(old, b'/static/js/joyai-client.js?v=scene-flow-restored-20260925'))
helper = etherea / 'scripts/smoke-tests/joyai-transition-live.cjs'
data = helper.read_bytes()
old = b'c.sceneMotionSupported=Number.isFinite(health.upscale?.gpu_video?.motion_history_frames);'
assert data.count(old) == 1
(out / 'joyai-transition-live.cjs.rejected').write_bytes(data)
helper.write_bytes(data.replace(old, b''))

# One-off adoption uses the established live-client diagnostic, releasing retained
# images only while presentation/control are idle. It never changes the prompt.
script = helper.read_text()
old = 'if(c){Object.setPrototypeOf(c,current.JoyAIClient.prototype);'
assert script.count(old) == 1
new = '''if(c){
     const deadline=Date.now()+30000;
     while(c.playing||c.transitioning||c.scenePreparing||c.controlTask){
      if(Date.now()>deadline)throw Error('Scene still transitioning; restore adoption deferred');
      await new Promise(r=>setTimeout(r,20));
     }
     window.__generatedMotionTest?.cleanup();
     c.clearMotionState?.();c.sceneMotionSupported=false;
     Object.setPrototypeOf(c,current.JoyAIClient.prototype);'''
script = script.replace(old, new).replace("const fs=require('node:fs'), W=require('ws');", "const fs=require('node:fs'), W=require(" + json.dumps(str(etherea / 'node_modules/ws')) + ");")
(out / 'adopt-restored-client.cjs').write_text(script)
(root / 'tulip/restored-handoff-current.txt').write_text(str(out))
(out / 'restored-source-hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [x[0] for x in pairs]+[manager,helper]}, indent=2))
print(out)
