"""Summarize settled windows from measure_padded_prompt.cjs, without live changes."""
import argparse
import json
from pathlib import Path
import statistics

p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);args=p.parse_args()
source=json.loads(args.input.read_text())
def stats(values):
    return {'count':len(values),'median':statistics.median(values),'mean':statistics.mean(values),
            'p95':sorted(values)[min(len(values)-1,int(len(values)*.95))]} if values else {}
result={'complete':source['complete'],'error':source['error'],'restore':source['restore'],'phases':{}}
for phase in ('original','variant'):
    rows=[r for r in source['samples'] if r['phase']==phase and r['phaseSeconds']>=15]
    if len(rows)<2:continue
    first,last=rows[0],rows[-1];seen={};chunks={}
    for r in rows:
        for s in r['stage']['samples']:
            if first['at']<=s['at']<=last['at']:seen[(s['at'],s['stage'],s['chunk'])]=s
        chunks[r['profile']['chunk_idx']]=r['profile']
    stages={}
    for s in seen.values():
        label=s['stage']+':'+str(s.get('prompt_capacity',''))
        stages.setdefault(label,[]).append(s['gpu_ms'])
    delta={k:last['gpu'].get(k,0)-first['gpu'].get(k,0) for k in ('sent','dropped','underruns','restart_count')}
    delta['fps']=delta['sent']/(last['at']-first['at'])
    delta['newGaps']=[g for g in last['gpu'].get('delivery_gaps',[]) if g['at']>=first['at']]
    result['phases'][phase]={'seconds':last['at']-first['at'],
        'promptTokens':sorted({r['profile']['graph_prompt_tokens'] for r in rows}),
        'capacity':sorted({r['profile']['graph_prompt_capacity'] for r in rows}),
        'activeRows':sorted({r['profile'].get('graph_active_text_rows',r['profile']['graph_prompt_capacity']) for r in rows}),
        'stagesMs':{k:stats(v) for k,v in stages.items()},
        'serverChunkMs':stats([r['total_server_chunk_s']*1000 for r in chunks.values()]),
        'previewMs':stats([r['stats']['captureToCanvasMs'] for r in rows if r.get('stats',{}).get('captureToCanvasMs')]),
        'delivery':delta,'errors':[r['gpu']['error'] for r in rows if r['gpu'].get('error')],
        'memoryBefore':first['memory'],'memoryAfter':last['memory']}
args.input.with_suffix('.summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
