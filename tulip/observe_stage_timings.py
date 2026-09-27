"""Read-only, bounded collection of stage spans and live delivery/allocator metrics."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('output', type=Path)
parser.add_argument('--seconds', type=int, default=90)
args = parser.parse_args()
if not 10 <= args.seconds <= 300:
    parser.error('seconds must be between 10 and 300')
if args.output.exists():
    raise RuntimeError('Evidence already exists')

def get(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)

def stats(values):
    return {'count': len(values), 'median': statistics.median(values),
            'mean': statistics.mean(values), 'p95': sorted(values)[min(len(values)-1, int(len(values)*.95))],
            'max': max(values)} if values else {}

rows = []
seen = {}
chunks = {}
started = time.monotonic()
with args.output.open('x', encoding='utf-8') as output:
    while time.monotonic() - started < args.seconds:
        at = time.time()
        debug = get('http://127.0.0.1:7873/debug')
        health = get('http://127.0.0.1:7873/health')
        live = get('http://127.0.0.1:8793/control/multistream/list')
        tile = next((s for s in live['streams'] if s['type'] == 'joyai'), {})
        stage = debug.get('stage_timings') or {}
        for row in stage.get('samples', []):
            if row['at'] >= (rows[0]['at'] if rows else at):
                seen[(row['at'], row['stage'], row['chunk'])] = row
        session = debug.get('session') or {}
        profile = session.get('last_chunk_profile') or {}
        if profile:
            chunks[(tile.get('id'), profile.get('chunk_idx'))] = profile
        gpu = health.get('upscale', {}).get('gpu_video', {})
        smi = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        row = {'at': at, 'stage': stage, 'profile': profile, 'memory': debug.get('memory'),
               'gpu_video': gpu, 'upscale': health.get('upscale'), 'stats': tile.get('runtime', {}).get('stats'),
               'queues': session.get('queues'), 'devices': session.get('devices'),
               'nvidia_smi': smi.stdout.strip(), 'smi_error': smi.stderr.strip()}
        rows.append(row)
        output.write(json.dumps(row) + '\n'); output.flush()
        if len(rows) % 10 == 0:
            print(json.dumps({'seconds': round(time.monotonic()-started), 'samples': len(seen),
                              'sent': gpu.get('sent'), 'underruns': gpu.get('underruns')}), flush=True)
        time.sleep(2)

first, last = rows[0], rows[-1]
elapsed = last['at'] - first['at']
grouped = {}
for row in seen.values():
    label = row['stage'] + '@' + row['device']
    if row.get('prompt_capacity') is not None:
        label += ':' + str(row['prompt_capacity'])
    grouped.setdefault(label, []).append(row['gpu_ms'])
delivery = {key: last['gpu_video'].get(key, 0)-first['gpu_video'].get(key, 0)
            for key in ('sent', 'dropped', 'underruns', 'restart_count')}
delivery['fps'] = delivery['sent']/elapsed
delivery['new_gaps'] = [g for g in last['gpu_video'].get('delivery_gaps', []) if g['at'] >= first['at']]
summary = {'elapsed_s': elapsed, 'stages_ms': {key: stats(values) for key, values in grouped.items()},
           'server_chunk_ms': stats([p['total_server_chunk_s']*1000 for p in chunks.values() if 'total_server_chunk_s' in p]),
           'preview_latency_ms': stats([r['stats']['frameTiming']['latencyMs'] for r in rows if r.get('stats', {}).get('frameTiming', {}).get('latencyMs')]),
           'delivery': delivery, 'memory_before': first['memory'], 'memory_after': last['memory'],
           'devices': last['devices'], 'stage_error': last['stage'].get('error'),
           'note': 'CUDA stream spans include launch gaps and contention; stages overlap and must not be summed.'}
args.output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary), flush=True)
