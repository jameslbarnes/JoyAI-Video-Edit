"""Exercise the explicit handoff switch without replacing the live session."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('report', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
samples = []
def get(endpoint):
    with urlopen('http://127.0.0.1:7873/' + endpoint, timeout=5) as response:
        return json.load(response)

for mode in ('on', 'off', 'on'):
    subprocess.run([sys.executable, str(root / 'tulip/set_rife_gpu.py'), mode], check=True)
    phase = []
    for i in range(16):
        health, debug = get('health'), get('debug')
        sr = health['upscale']
        state = {'at': time.time(), 'mode': mode, 'elapsed': i, 'rife_gpu': sr['rife_gpu_requested'],
                 'handoff': sr['last'].get('rife_gpu_handoff'), 'srSeconds': sr['last'].get('total_s'),
                 'gpu': sr['gpu_video'], 'error': sr['error'],
                 'connectedAt': debug['ws']['connected_at'], 'memory': debug['memory']['devices']}
        phase.append(state); samples.append(state)
        args.report.write_text(json.dumps({'samples': samples}, indent=2))
        time.sleep(1)
    steady = phase[5:]
    assert all(s['rife_gpu'] == (mode == 'on') and s['handoff']['gpu_input'] == (mode == 'on') for s in steady)
    assert all(not s['error'] and not s['gpu']['error'] and s['handoff']['input_download_bytes'] == 0 for s in steady)
    assert steady[-1]['gpu']['sent'] - steady[0]['gpu']['sent'] > 200
assert len({s['connectedAt'] for s in samples}) == 1, 'Live session was replaced'
assert len({s['gpu']['mux_pid'] for s in samples}) == 1, 'Publisher was restarted'
assert samples[-1]['gpu']['underruns'] == samples[0]['gpu']['underruns']
report = {'ok': True, 'sameSession': True, 'samePublisher': True, 'finalMode': 'on',
          'samples': samples}
args.report.write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != 'samples'}), flush=True)
