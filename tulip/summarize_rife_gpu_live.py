"""Read-only live acceptance summary for the GPU-resident RIFE handoff."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from urllib.request import urlopen

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('evidence', type=Path)
args = parser.parse_args()
out = args.evidence
def read(name): return json.loads((out / name).read_text(encoding='utf-8-sig'))
def get(url):
    with urlopen(url, timeout=10) as response: return json.load(response)
def save(name, value): (out / name).write_text(json.dumps(value, indent=2), encoding='utf-8')
health = get('http://127.0.0.1:7873/health')
debug = get('http://127.0.0.1:7873/debug')
live = get('http://127.0.0.1:8793/control/multistream/list')
for name, value in [('health', health), ('debug', debug), ('live', live)]: save('final-' + name + '.json', value)
before = next(s for s in read('deploy-before-live.json')['streams'] if s['type'] == 'joyai')['runtime']['stats']
tile = next(s for s in live['streams'] if s['type'] == 'joyai')
now = tile['runtime']['stats']
whep = [json.loads(line) for line in (out / 'whep-verified.jsonl').read_text().splitlines()]
rows = [r for r in whep if r.get('event') == 'stats' and r.get('inboundVideo')]
seconds = (datetime.fromisoformat(rows[-1]['ts']) - datetime.fromisoformat(rows[0]['ts'])).total_seconds()
a, b = rows[0]['inboundVideo'], rows[-1]['inboundVideo']
scene = [json.loads(line) for line in (out / 'scene-transition.jsonl').read_text().splitlines()]
frames = next(r['frames'] for r in scene if r.get('event') == 'frames')
scene_stats = [r['data']['stats'] for r in scene if r.get('data', {}).get('stats')]
toggle = read('toggle-verification.json')
gpu = health['upscale']['gpu_video']
summary = {
    'status': tile['status'], 'native': [now['nativeWidth'], now['nativeHeight'], now['nativeFps']],
    'rifeMultiplier': now['rifeMultiplier'], 'backend': health['upscale']['backend'],
    'gpuOutput': gpu['geometry'], 'gpuHandoff': health['upscale']['last']['rife_gpu_handoff'],
    'promptPreserved': now['prompt'] == before['prompt'],
    'strengthPreserved': now['strength']['target'] == before['strength']['target'],
    'baseline': read('baseline-observe.json')['summary'],
    'promptChanges': read('prompt-observe.json')['summary'],
    'steady': read('steady-observe.json')['summary'],
    'whep': {'seconds': seconds, 'decodedFps': (b['framesDecoded']-a['framesDecoded']) / seconds,
             'freezes': b['freezeCount'], 'dropped': b['framesDropped'], 'packetsLost': b['packetsLost'],
             'resolution': [b['frameWidth'], b['frameHeight']]},
    'manualToggle': {k: v for k, v in toggle.items() if k != 'samples'},
    'scene': {'observedFrames': len(frames), 'presentation': dict(Counter(f['presentation'] for f in frames)),
              'epochs': sorted(set(f['epoch'] for f in frames)),
              'clientErrors': [s.get('firstError') for s in scene_stats if s.get('firstError')]},
    'publisher': {k: gpu.get(k) for k in ('mux_pid', 'restart_count', 'dropped', 'underruns', 'error', 'delivery_gaps')},
    'memory': [{k: d[k] for k in ('physicalGpu', 'allocationRetries', 'outOfMemoryCount')} for d in debug['memory']['devices']],
    'note': 'Initial probe with an unreachable local STUN endpoint failed ICE; corrected probe uses the previously working STUN setting. No renderer change was required.'}
save('final-summary.json', summary)
print(json.dumps(summary), flush=True)
