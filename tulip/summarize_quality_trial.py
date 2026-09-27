"""Capture read-only qualification evidence and concise measured throughput."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from urllib.request import urlopen

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
args = parser.parse_args()
out = args.directory.resolve(strict=True)

def read(name):
    return json.loads((out / name).read_text(encoding='utf-8-sig'))

def get(url):
    with urlopen(url, timeout=10) as response:
        return json.loads(response.read())

def save(name, value):
    (out / name).write_text(json.dumps(value, indent=2), encoding='utf-8')

health = get('http://127.0.0.1:7873/health')
debug = get('http://127.0.0.1:7873/debug')
live = get('http://127.0.0.1:8793/control/multistream/list')
for name, value in [('final-health.json', health), ('final-debug.json', debug), ('final-live.json', live)]:
    save(name, value)
tile = next(t for t in live['streams'] if t['type'] == 'joyai')
before = next(t for t in read('before-live.json')['streams'] if t['type'] == 'joyai')['runtime']['stats']
now = tile['runtime']['stats']
summary = dict(backend=health['upscale']['backend'], status=tile['status'],
    originalPromptPreserved=now['prompt'] == before['prompt'],
    strengthPreserved=now['strength']['target'] == before['strength']['target'],
    memory=[{k: d[k] for k in ('physicalGpu', 'allocationRetries', 'outOfMemoryCount')}
            for d in debug['memory']['devices']])
for name in ('observe', 'stress'):
    if not (out / (name + '.json')).exists():
        continue
    measured = dict(read(name + '.json')['summary'])
    measured['gapCount'] = len(measured.pop('gaps', []))
    summary[name] = measured
for name in ('whep', 'whep-stress', 'whep-hardware', 'whep-software'):
    path = out / (name + '.jsonl')
    if not path.exists():
        continue
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()]
    rows = [r for r in rows if r.get('event') == 'stats' and r.get('inboundVideo')]
    if len(rows) < 2:
        raise RuntimeError('Insufficient WebRTC samples: ' + name)
    seconds = (datetime.fromisoformat(rows[-1]['ts']) - datetime.fromisoformat(rows[0]['ts'])).total_seconds()
    a, b = rows[0]['inboundVideo'], rows[-1]['inboundVideo']
    summary[name] = dict(seconds=seconds, decodedFps=(b['framesDecoded']-a['framesDecoded'])/seconds,
        dropped=b.get('framesDropped'), freezes=b.get('freezeCount'),
        freezeMilliseconds=b.get('totalFreezesDurationMs'), packetsLost=b.get('packetsLost'))
save('final-summary.json', summary)
print(json.dumps(summary))
