"""Explicit restoration of the saved 648p Real-ESRGAN favorite. Default: verify only."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FAVORITE = ROOT / 'tulip/favorites/648p-realesrgan-20260922'

def verified_files():
    manifest = json.loads((FAVORITE / 'manifest.json').read_text(encoding='utf-8'))
    result = []
    for entry in manifest['files']:
        target = (ROOT / entry['path']).resolve()
        source = (FAVORITE / 'files' / entry['path']).resolve()
        if not target.is_relative_to(ROOT) or not source.is_relative_to((FAVORITE / 'files').resolve()):
            raise ValueError('Snapshot path escapes its root')
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('Snapshot checksum mismatch: ' + entry['path'])
        result.append((target, data))
    scene_file = FAVORITE / 'scene.json'
    if hashlib.sha256(scene_file.read_bytes()).hexdigest() != manifest['sceneSha256']:
        raise ValueError('Saved scene checksum mismatch')
    return manifest, result

def main():
    global FAVORITE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--favorite', default=FAVORITE.name, help='Saved directory name under tulip/favorites')
    parser.add_argument('--apply', action='store_true', help='Back up current files, restore saved runtime source/settings, restart only JoyAI')
    parser.add_argument('--saved-scene', action='store_true', help='Also restore the saved JoyAI prompt and strength; shader animation is not rewound')
    args = parser.parse_args()
    selected = (ROOT / 'tulip/favorites' / args.favorite).resolve()
    if selected.parent != (ROOT / 'tulip/favorites').resolve():
        raise ValueError('Favorite must be a direct child of tulip/favorites')
    FAVORITE = selected
    manifest, files = verified_files()
    print(json.dumps({'favorite':manifest['name'], 'verifiedFiles':len(files), 'apply':args.apply}), flush=True)
    if not args.apply:
        return
    # Reuse the captured SDK-owned lifecycle helpers, not a later edited version.
    helper = ROOT / 'tulip/native600_trial.py'
    favorite_helper = FAVORITE / 'files/tulip/quality_trial_lifecycle.py'
    code = (FAVORITE / 'files/tulip/native600_trial.py').read_text(encoding='utf-8').split('from easel_agent.ops_guard import hold_status')[0]
    code = code.replace('helper.read_text()', "favorite_helper.read_text(encoding='utf-8')")
    ns = {'__file__':str(helper), '__name__':'favorite_restore_bootstrap', 'favorite_helper':favorite_helper}
    exec(compile(code,str(helper),'exec'),ns)
    from easel_agent.ops_guard import hold_status
    if hold_status().get('held'):
        raise RuntimeError('Another operations hold is active; nothing changed')
    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = ROOT / 'tulip/evidence' / ('favorite-restore-' + stamp)
    out.mkdir(parents=True,exist_ok=False); ns['out'] = out
    listing,tile = ns['current']()
    scene = json.loads((FAVORITE/'scene.json').read_text(encoding='utf-8'))
    if not args.saved_scene and tile:
        stats = tile.get('runtime',{}).get('stats',{})
        scene['prompt'] = stats.get('prompt') or scene['prompt']
        strength = stats.get('strength')
        if isinstance(strength,dict):
            scene['strength'] = strength.get('target',strength.get('value',scene['strength']))
    ns['save']('before-live',listing)
    with zipfile.ZipFile(out/'before-files.zip','w',zipfile.ZIP_DEFLATED) as z:
        for target,_ in files:
            if target.exists(): z.write(target,target.relative_to(ROOT).as_posix())
    ns['act']('ops.hold',{'ttlMinutes':25,'reason':'Explicit user restore of saved 648p Real-ESRGAN favorite'})
    try:
        ns['stop']()
        for target,data in files:
            target.parent.mkdir(parents=True,exist_ok=True)
            pending = target.with_name(target.name+'.favorite-pending')
            pending.write_bytes(data);pending.replace(target)
        cfg=json.loads((ROOT/'tulip/native-quality.json').read_text())
        ns['launch'](cfg,scene['prompt'],scene['strength'],'favorite-restored')
    finally:
        # Report any failure without automatically switching quality or stopping a live feed.
        ns['act']('ops.resume')

if __name__ == '__main__':
    main()
