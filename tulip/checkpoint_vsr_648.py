"""Preserve the qualified 648p VSR runtime before further native-resolution trials."""
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parents[1]
out = root/'tulip/favorites/648p-vsr-20260922'
out.mkdir(exist_ok=False)
manifest = json.loads((root/'tulip/favorites/648p-realesrgan-20260922/manifest.json').read_text())
def get(url):
    with urlopen(url, timeout=10) as r: return json.loads(r.read())
health = get('http://127.0.0.1:7873/health')
live = get('http://127.0.0.1:8793/control/multistream/list')
tile = next(t for t in live['streams'] if t['type']=='joyai')
assert tile['status']=='live' and health['upscale']['backend']=='vsr_ultra'
config = json.loads((root/'tulip/native-quality.json').read_text())
assert config=={'width':1152,'height':648,'fps':12,'rife_multiplier':2}
paths = {entry['path'] for entry in manifest['files']}
paths.update(p.relative_to(root).as_posix() for p in (root/'deploy/xvideo').rglob('*.py'))
paths.update(['tulip/select_upscaler.py','tulip/verify_vsr_output.py','tulip/restore_favorite.py'])
entries=[]
for relative in sorted(paths):
    data=(root/relative).read_bytes()
    target=out/'files'/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(data)
    entries.append(dict(path=relative,sha256=hashlib.sha256(data).hexdigest()))
stats=tile['runtime']['stats']
scene=json.dumps(dict(prompt=stats['prompt'],strength=stats['strength']['target']),indent=2).encode()
(out/'scene.json').write_bytes(scene)
(out/'live.json').write_text(json.dumps(dict(health=health,live=live),indent=2),encoding='utf-8')
manifest.update(name='648p - VSR Ultra qualified 24fps',backend='vsr_ultra',
    createdUtc=datetime.now(timezone.utc).isoformat(),files=entries,
    sceneSha256=hashlib.sha256(scene).hexdigest())
(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
(out/'README.md').write_text('Qualified 648p/VSR Ultra, 12 native + RIFE 2, 1440p GPU output.\n'
    'Verify: .venv/Scripts/python.exe tulip/restore_favorite.py --favorite 648p-vsr-20260922\n'
    'Explicit restore: append --apply. Keeps current prompt/strength by default.\n'
    'Uses the installed model weights and pinned local NVIDIA package, not a portable installation.\n',encoding='utf-8')
with zipfile.ZipFile(out.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
    for path in out.rglob('*'):
        if path.is_file(): archive.write(path,path.relative_to(out))
print(json.dumps(dict(snapshot=str(out),files=len(entries))))
