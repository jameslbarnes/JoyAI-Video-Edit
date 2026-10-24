"""Bounded isolated 1440p NDI test. Never connects to the live ingest port.

Starts its own bridge and hidden GL receiver; all children are reaped in finally.
Uses captured high-detail stills as encoder inputs, not an extra model instance.
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import cv2
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'deploy'))
from xvideo.serving.gpu_video_output import GpuFrame
from xvideo.serving.gpu_ndi_output import GpuNdiPublisher
from xvideo.serving.ndi_pixels import pack_uyvy


def main():
    evidence=Path(sys.argv[1]);evidence.mkdir(parents=True,exist_ok=True)
    sdk=ROOT.parent/'etherea-ai'
    env={**os.environ,'NDI_BRIDGE_PORT':'17879','NDI_BRIDGE_FRAME_PORT':'17880'}
    # Refuse to collide with any operator-owned process.
    for port in (17879,17880):
        with socket.socket() as check: check.bind(('127.0.0.1',port))
    children=[];publisher=None;log=(evidence/'bridge.log').open('wb')
    try:
        bridge=subprocess.Popen([r'C:\Program Files\nodejs\node.exe',
            str(sdk/'scripts/streaming/ndi-bridge.js'),'17879'],cwd=sdk,env=env,
            stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        children.append(bridge)
        for _ in range(50):
            try:
                with socket.create_connection(('127.0.0.1',17880),timeout=.2):break
            except OSError:
                if bridge.poll() is not None:raise RuntimeError('Isolated NDI bridge exited')
                time.sleep(.1)
        else:raise RuntimeError('Isolated NDI bridge did not listen')
        torch.cuda.set_per_process_memory_fraction(.03,0)
        frames=[]
        base=Path(os.environ['LOCALAPPDATA'])/'etherea/quality/20261019-2048'
        for index in (1,2,3):
            rgb=cv2.cvtColor(cv2.imread(str(base/f'local-ndi-{index:02d}.png')),cv2.COLOR_BGR2RGB)
            rgb=cv2.resize(rgb,(2592,1440))
            frames.append(GpuFrame.record(torch.from_numpy(rgb).to('cuda:0')))
        pack_uyvy(frames[0].wait()).cpu()  # compilation and CUDA initialization outside measurements
        name=f'ETHEREA TEST optimized {os.getpid()}'
        publisher=GpuNdiPublisher('cuda:0',port=17880,name=name)
        publisher.submit([frames[i%3] for i in range(16)],24)
        recvLog=(evidence/'receiver.log').open('wb')
        receiver=subprocess.Popen([str(ROOT.parent/'easel/build-fresh/Release/test_ndi_gpu_receive.exe'),
            f'TULIP ({name})','34'],stdout=recvLog,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW)
        children.append(receiver)
        began=time.perf_counter();deadline=began;nextSample=began+3;samples=[];i=0
        while time.perf_counter()-began<36:
            publisher.submit([frames[i%3]],24);i+=1
            now=time.perf_counter()
            if now>=nextSample:
                sample={'at':time.time(),**publisher.status()};samples.append(sample)
                print(json.dumps(sample),flush=True);nextSample+=5
            deadline+=1/24
            time.sleep(max(0,deadline-time.perf_counter()))
            if publisher.status()['error']:raise RuntimeError(publisher.status()['error'])
        publisher.close()
        receiver.wait(timeout=5);recvLog.close()
        if receiver.returncode:raise RuntimeError('GPU receiver failed: '+(evidence/'receiver.log').read_text())
        a,b=samples[0],samples[-1]
        result={'seconds':b['last_send_at']-a['last_send_at'],
            'fps':(b['sent']-a['sent'])/(b['last_send_at']-a['last_send_at']),
            'dropped':b['dropped']-a['dropped'],'underruns':b['underruns']-a['underruns'],
            'rawBytesPerFrame':(b['raw_download_bytes']-a['raw_download_bytes'])/(b['sent']-a['sent']),
            'samples':samples,'receiverLog':(evidence/'receiver.log').read_text()}
        (evidence/'benchmark.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k!='samples'}),flush=True)
    finally:
        if publisher is not None:publisher.close()
        for child in reversed(children):
            if child.poll() is None:child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        log.close()


if __name__=='__main__':main()
