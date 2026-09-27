"""Explicit live interpolation selection. No renderer restart or automatic fallback."""
import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('backend',choices=('rife','dlssg'));args=p.parse_args()
root=Path(__file__).resolve().parents[1]
presentation=json.loads((root/'tulip/output-presentation.json').read_text(encoding='utf-8-sig'))
if args.backend=='dlssg' and not (presentation.get('backend')=='vsr_ultra' and presentation.get('rife_gpu') and presentation.get('progressive')):
    raise RuntimeError('DLSSG requires the existing GPU VSR progressive output path')
path=root/'tulip/interpolation-settings.json';pending=path.with_suffix('.pending.json')
pending.write_text(json.dumps({'backend':args.backend},indent=2)+'\n');pending.replace(path)
print(json.dumps({'requested':args.backend,'takesEffect':'next interpolation chunk on a deployed service','automaticRollback':False}))
