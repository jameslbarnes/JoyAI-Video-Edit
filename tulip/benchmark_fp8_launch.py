"""Low-duty FP8 launch probe on the output GPU; never changes the live renderer.

The production kernel, FP8 inputs, BK=128 accumulation and output precision
remain identical. Only tile size, warps and pipeline stages are compared.
Only bit-identical candidates are eligible. Live publisher regressions stop
this probe, never JoyAI. Timings under concurrent live load are indicative.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('evidence', type=Path)
parser.add_argument('--physical-gpu', type=int, default=0)
parser.add_argument('--rows', type=int, nargs='+', default=[512, 1024, 1296, 2592])
parser.add_argument('--repeats', type=int, default=7)
parser.add_argument('--cold-cache', action='store_true')
parser.add_argument('--config', action='append', help='BM,BN,warps,stages (repeatable)')
parser.add_argument('--projection', action='append', choices=('qkv','projection','mlp_up','mlp_down'))
args = parser.parse_args()
if args.physical_gpu != 0:
    parser.error('This live probe is restricted to physical output GPU0')
if any(m not in (512, 1024, 1296, 2592) for m in args.rows) or not 3 <= args.repeats <= 15:
    parser.error('Use 648p production row counts and 3..15 repeats')
args.evidence.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'deploy'))
os.environ.update(CUDA_VISIBLE_DEVICES=str(args.physical_gpu), JOYOMNI_FP8_FAST_ACCUM='1',
                  CUDA_HOME=str(root / '.toolchain/Library'),
                  TRITON_CACHE_DIR=str(root / 'tulip/cache/fp8-launch-probe'),
                  MAX_JOBS='1', TORCHINDUCTOR_COMPILE_THREADS='1')
os.environ['PATH'] = str(root / '.toolchain/Library/bin') + os.pathsep + os.environ['PATH']

def get(endpoint):
    with urllib.request.urlopen('http://127.0.0.1:7873/' + endpoint, timeout=5) as r:
        return json.load(r)

before_health, before_debug = get('health'), get('debug')
for name, value in [('before-health', before_health), ('before-debug', before_debug)]:
    (args.evidence / (name + '.json')).write_text(json.dumps(value, indent=2))
initial_gpu = before_health['upscale']['gpu_video']
if not before_health['runtime_loaded'] or initial_gpu.get('error'):
    raise RuntimeError('Expected healthy live JoyAI')
free = float(subprocess.check_output(['nvidia-smi', '-i', str(args.physical_gpu),
             '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
if free < 3500:
    raise RuntimeError('Need 3.5 GiB free before starting the bounded probe')

import torch
import triton
from xvideo.models.dit.fp8_linear import _fa_mm_kernel, _fa_pick_config

torch.set_num_threads(1)
torch.manual_seed(20260923)
torch.cuda.set_per_process_memory_fraction(.045)
device = torch.device('cuda:0')
stream = torch.cuda.Stream(device=device, priority=0)
configs = [(64,128,4,3), (64,128,8,3), (64,256,4,3), (64,256,8,3),
           (128,128,4,3), (128,128,8,3), (128,256,8,3), (128,256,8,4),
           (64,256,8,4), (128,128,4,4)]
projections = [('qkv', 4096, 12288), ('projection', 4096, 4096),
               ('mlp_up', 4096, 16384), ('mlp_down', 16384, 4096)]
if args.config:
    configs = [tuple(int(n) for n in value.split(',')) for value in args.config]
    if any(len(c)!=4 or c[0] not in (32,64,128) or c[1] not in (64,128,256)
           or c[2] not in (4,8,16) or c[3] not in (1,2,3,4) for c in configs):
        parser.error('Unsupported launch configuration')
if args.projection:
    projections = [p for p in projections if p[0] in args.projection]
flush_buffer = torch.empty(256*1024*1024,device=device,dtype=torch.uint8) if args.cold_cache else None

def live_guard():
    current = get('health')['upscale']['gpu_video']
    if (current.get('error') or current.get('underruns', 0) > initial_gpu.get('underruns', 0)
            or current.get('dropped', 0) > initial_gpu.get('dropped', 0)
            or current.get('mux_pid') != initial_gpu.get('mux_pid')
            or current.get('restart_count', 0) != initial_gpu.get('restart_count', 0)):
        raise RuntimeError('Publisher changed/regressed; stop only the offline probe')

def launch(config):
    bm, bn, warps, stages = config
    _fa_mm_kernel[(triton.cdiv(M,bm)*triton.cdiv(N,bn),)](
        a, b, output, xs, ws, bias, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        output.stride(0), output.stride(1), HAS_BIAS=True,
        BM=bm, BN=bn, BK=128, GROUP=8, num_warps=warps, num_stages=stages)

def capture(config):
    launch(config)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream, capture_error_mode='thread_local'):
        launch(config)
    return graph

def timed(graph):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        if flush_buffer is not None: flush_buffer.zero_()
        start.record(stream); graph.replay(); end.record(stream)
    end.synchronize()
    result = start.elapsed_time(end)
    time.sleep(.04)
    return result

reports = []
try:
    with torch.no_grad(), torch.cuda.stream(stream), (args.evidence/'results.jsonl').open('x') as file:
        for M in args.rows:
            for name, K, N in projections:
                live_guard()
                # Real production FP8 layout and scaling; synthetic inputs held
                # fixed across candidates, not a model-quality qualification.
                a = torch.randint(-28,29,(M,K),device=device,dtype=torch.int8).to(torch.float8_e4m3fn)
                b = torch.randint(-28,29,(N,K),device=device,dtype=torch.int8).to(torch.float8_e4m3fn).t()
                xs = torch.rand(M,device=device,dtype=torch.float32)*.01+.001
                ws = torch.rand(N,device=device,dtype=torch.float32)*.01+.001
                bias = torch.randn(N,device=device,dtype=torch.bfloat16)*.1
                output = torch.empty((M,N),device=device,dtype=torch.bfloat16)
                base_config = _fa_pick_config(M,N,K)
                base_graph = capture(base_config)
                expected = output.clone()
                assert torch.isfinite(expected).all()
                results = []
                for config in dict.fromkeys([tuple(base_config), *configs]):
                    live_guard()
                    graph = None
                    try:
                        graph = capture(config)
                        exact = torch.equal(output, expected)
                        base_ms, candidate_ms = [], []
                        if exact:
                            for i in range(args.repeats):
                                if i % 2:
                                    candidate_ms.append(timed(graph)); base_ms.append(timed(base_graph))
                                else:
                                    base_ms.append(timed(base_graph)); candidate_ms.append(timed(graph))
                        result = {'config': config, 'bitExact': exact, 'baselineMs': base_ms,
                                  'candidateMs': candidate_ms,
                                  'pairedRatioMedian': statistics.median(c/b for c,b in zip(candidate_ms,base_ms)) if exact else None}
                    except torch.cuda.OutOfMemoryError:
                        raise
                    except Exception as exc:
                        result = {'config': config, 'error': repr(exc)}
                    results.append(result)
                    if graph is not None: del graph
                    time.sleep(.1)
                eligible = [r for r in results if r.get('bitExact') and r.get('candidateMs')]
                best = min(eligible,key=lambda r:r['pairedRatioMedian']) if eligible else None
                row = {'projection': name,'M':M,'N':N,'K':K,'baselineConfig':base_config,
                       'results':results,'best':best,'allocatedMiB':torch.cuda.memory_allocated()/2**20,
                       'reservedMiB':torch.cuda.memory_reserved()/2**20}
                reports.append(row);file.write(json.dumps(row)+'\n');file.flush()
                print(json.dumps({k:row[k] for k in ['projection','M','N','K','baselineConfig','best','reservedMiB']}),flush=True)
                del base_graph, expected, output, a, b, xs, ws, bias
                stream.synchronize()
                torch.cuda.empty_cache()
                time.sleep(.2)
    live_guard()
finally:
    summary = {'device':torch.cuda.get_device_name(), 'physicalGpu':args.physical_gpu,
               'peakAllocatedMiB':torch.cuda.max_memory_allocated()/2**20,
               'peakReservedMiB':torch.cuda.max_memory_reserved()/2**20, 'reports':reports,
               'coldCache':args.cold_cache,
               'note':'Concurrent-live microbenchmark only; no renderer settings/weights/precision changed.'}
    (args.evidence/'summary.json').write_text(json.dumps(summary,indent=2))
    for endpoint in ['health','debug']:
        try:(args.evidence/('after-'+endpoint+'.json')).write_text(json.dumps(get(endpoint),indent=2))
        except Exception:pass
