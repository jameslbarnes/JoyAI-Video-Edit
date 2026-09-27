"""Compare fixed encoded outputs on an isolated local RTMP/WHEP route.

This measures receiver/delivery behavior, not interpolation compute throughput.
Also observes the unchanged live DLSSG route. Owns and stops only its publishers.
"""
from pathlib import Path
import argparse,json,subprocess,time,urllib.request
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def health():
    return json.load(urllib.request.urlopen('http://127.0.0.1:7873/health',timeout=10))

def summarize(path):
    records=[json.loads(x) for x in path.read_text().splitlines()]
    stats=[r for r in records if r.get('event')=='stats' and r.get('inboundVideo')]
    frames=[f for r in records if r.get('event')=='frame_timing' for f in r['frames']]
    assert len(stats)>5 and len(frames)>100, 'No usable receiver data'
    # Remove receiver startup only; retain all subsequent stalls and cuts.
    start=frames[0]['callbackMs']+5000
    frames=[f for f in frames if f['callbackMs']>=start]
    stats=[s for s in stats if s['receiverTimestampMs']>=start]
    first,last=stats[0],stats[-1];a,b=first['inboundVideo'],last['inboundVideo']
    seconds=(last['receiverTimestampMs']-first['receiverTimestampMs'])/1000
    def percentile(values):
        return {'p50':float(np.percentile(values,50)),'p95':float(np.percentile(values,95)),
                'p99':float(np.percentile(values,99)),'max':float(max(values))}
    media=np.diff([f['mediaTime'] for f in frames])*1000
    display=np.diff([f['expectedDisplayTimeMs'] for f in frames])
    callbacks=np.diff([f['callbackMs'] for f in frames])
    presented=np.diff([f['presentedFrames'] for f in frames])
    return {'seconds':seconds,'decodedFps':(b['framesDecoded']-a['framesDecoded'])/seconds,
        'decodedFrames':b['framesDecoded']-a['framesDecoded'],
        'droppedFrames':b['framesDropped']-a['framesDropped'],
        'packetsLost':b['packetsLost']-a['packetsLost'],
        'freezeCount':b['freezeCount']-a['freezeCount'],
        'freezeDurationSeconds':b['totalFreezesDuration']-a['totalFreezesDuration'],
        'callbackSamples':len(frames),'callbackGapsMs':percentile(callbacks),
        'expectedDisplayGapsMs':percentile(display),'mediaStepsMs':percentile(media),
        'mediaJumpsOver1_5Frames':int(np.count_nonzero(media>62.5)),
        'mediaNonIncreasingSteps':int(np.count_nonzero(media<=0)),
        'presentedCounterSkippedCallbacks':int(np.maximum(0,presented-1).sum()),
        'consecutivePresentationGapsMs':percentile(display[presented==1]),
        'displayGapsOver2Frames':int(np.count_nonzero(display>1000/12)),
        'frameTimingOverflow':max(r.get('overflow',0) for r in records if r.get('event')=='frame_timing'),
        'geometry':[b['frameWidth'],b['frameHeight']],'decoder':b.get('decoderImplementation'),
        'scope':'Separate local headless Edge receiver, excluding first 5 seconds. Display timestamps are compositor estimates, not TV scanout; 60 Hz quantization is expected.'}

def run(out,seconds):
    ff=ROOT.parent/'easel/external/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe'
    eth=ROOT.parent/'etherea-ai';edge=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
    save=lambda n,x:(out/n).write_text(json.dumps(x,indent=2),encoding='utf-8')
    save('receiver-live-health-before.json',health())
    report={}
    # Stable route differs from the live publisher and is used sequentially.
    for label in ['rife','dlssg','live-dlssg']:
        publisher=None
        with (out/(label+'-publisher.log')).open('x') as plog:
            route='joyai-gpu' if label=='live-dlssg' else 'interpolation-quality-eval'
            try:
                if label!='live-dlssg':
                    publisher=subprocess.Popen([str(ff),'-hide_banner','-loglevel','warning','-re','-stream_loop','-1','-i',str(out/(label+'-vsr1440.mp4')),
                        '-an','-c:v','copy','-f','flv','rtmp://127.0.0.1:1935/'+route],stdout=plog,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                    time.sleep(2)
                    assert publisher.poll() is None, 'Isolated replay publisher failed'
                file=out/(label+'-receiver.jsonl')
                with (out/(label+'-receiver.log')).open('x') as log:
                    subprocess.run(['node','scripts/streaming/whep-stats-probe.js','--whep-url',f'http://127.0.0.1:8889/{route}/whep',
                        '--seconds',str(seconds),'--no-turn','--frame-timing','--browser-executable',edge,'--out',str(file)],cwd=eth,
                        stdout=log,stderr=subprocess.STDOUT,check=True,timeout=seconds+45,creationflags=subprocess.CREATE_NO_WINDOW)
                report[label]=summarize(file);save('receiver-summary.json',report)
                print(json.dumps({'receiver':label,**report[label]}),flush=True)
            finally:
                if publisher is not None:
                    publisher.terminate()
                    try:publisher.wait(timeout=10)
                    except subprocess.TimeoutExpired:publisher.kill();publisher.wait(timeout=5)
    save('receiver-live-health-after.json',health())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=int,default=65);a=p.parse_args()
    run(a.out.resolve(),a.seconds)
