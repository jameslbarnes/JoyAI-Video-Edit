"""30s isolated live NDI SR proof. No app routing, prompt or model mutations.

Exactly one replaceable input slot; overload drops stale frames. Temporary output
sender/receivers are owned by this process and released on shutdown. Local only.
"""
import argparse
import json
import os
import struct
import threading
import time
from pathlib import Path

import numpy as np
import cv2
import torch
from PIL import Image
from websockets.sync.client import connect
from upscale_trial import Upscaler, FluxLatentUpscaler, live_status


class ResizeBaseline:
    def process(self, frame):
        started=time.perf_counter()
        result=cv2.resize(frame,(1296,720),interpolation=cv2.INTER_LANCZOS4)
        return result,{'pipeline_ms':(time.perf_counter()-started)*1000,'gpu_model_ms':0}


def bridge_request(socket, request_id, action, params):
    socket.send(json.dumps({'id':request_id,'action':action,'params':params}))
    deadline=time.perf_counter()+5
    while time.perf_counter()<deadline:
        message=socket.recv(timeout=2)
        if not isinstance(message,str):continue
        reply=json.loads(message)
        if reply.get('id')!=request_id:continue  # initial hello is not an acknowledgement
        if reply.get('error'):raise RuntimeError(reply['error'])
        return reply.get('result',reply)
    raise TimeoutError('Bridge did not acknowledge '+action)


class LatestReceiver:
    def __init__(self, name, width, height, fps=32):
        self.socket=connect('ws://127.0.0.1:7779/?video=0',max_size=16*1024**2,max_queue=1,open_timeout=5,close_timeout=1)
        self.id='upscale-recv-'+str(os.getpid())+'-'+str(width)
        self.socket.send(json.dumps({'id':1,'action':'ndi_keyed_receive_start','params':{
            'receiverId':self.id,'sourceName':name,'maxWidth':width,'maxHeight':height,
            'fps':fps,'bandwidth':'highest'}}))
        self.lock=threading.Lock();self.latest=None;self.dropped=0;self.count=0
        self.times=[];self.error=None;self.running=True
        self.thread=threading.Thread(target=self.read,daemon=True);self.thread.start()

    def read(self):
        try:
            while self.running:
                try: message=self.socket.recv(timeout=1)
                except TimeoutError: continue
                if isinstance(message,str):
                    result=json.loads(message)
                    if result.get('error'):raise RuntimeError(result['error'])
                    continue
                if len(message)<10 or message[0]!=4:continue
                n=message[1]
                if message[2:2+n].decode()!=self.id:continue
                w,h=struct.unpack_from('<II',message,2+n)
                frame=np.frombuffer(message[10+n:],np.uint8).reshape(h,w,4)[:,:,:3].copy()
                now=time.perf_counter()
                with self.lock:
                    if self.latest is not None:self.dropped+=1
                    self.latest=(now,frame);self.count+=1;self.times.append(now)
                    if len(self.times)>4096:self.times.pop(0)
        except Exception as exc:
            if self.running:self.error=str(exc)

    def take(self):
        if self.error:raise RuntimeError(self.error)
        with self.lock:
            result=self.latest;self.latest=None;return result

    def close(self):
        self.running=False;self.socket.close();self.thread.join(timeout=2)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=30)
    parser.add_argument('--kind',choices=['pixel','flux-latent','resize'],default='pixel')
    args=parser.parse_args()
    if args.out.exists():raise ValueError('Use a NEW evidence directory')
    if not 5<=args.seconds<=60:raise ValueError('Trial duration must be 5..60s')
    args.out.mkdir(parents=True)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.12,0);torch.backends.cudnn.benchmark=True
    before=live_status()
    if not before['fluxPaused']:raise RuntimeError('Flux must remain paused for this trial')
    model={'pixel':lambda:Upscaler(.5),'flux-latent':FluxLatentUpscaler,'resize':ResizeBaseline}[args.kind]()
    source=LatestReceiver('TULIP (ETHEREA JoyAI)',648,360)
    publisher=None;observer=None
    sender='upscale-trial-'+str(os.getpid())
    name='ETHEREA Upscale Trial '+str(os.getpid())
    metrics=[];states=[];observed=0;sample_hashes=set()
    try:
        deadline=time.perf_counter()+8;first=None
        while first is None and time.perf_counter()<deadline:
            first=source.take();time.sleep(.002)
        if first is None:raise RuntimeError('No input frame')
        for _ in range(4):model.process(first[1])
        publisher=connect('ws://127.0.0.1:7779/?video=0',max_size=8*1024**2,close_timeout=1)
        bridge_request(publisher,1,'video_sender_start',{
            'senderId':sender,'name':name,'width':1296,'height':720,'fps':32})
        start=time.perf_counter();poll_at=start;observe_at=start+2
        source.take();source.dropped=0
        while time.perf_counter()-start<args.seconds:
            item=source.take()
            if item is None:time.sleep(.001);continue
            arrival,frame=item
            process_at=time.perf_counter();output,timing=model.process(frame)
            rgba=np.empty((720,1296,4),np.uint8);rgba[:,:,:3]=output;rgba[:,:,3]=255
            tag=sender.encode()
            packet=bytes([3,len(tag)])+struct.pack('<II',1296,720)+tag+rgba.tobytes()
            send_at=time.perf_counter();publisher.send(packet);done=time.perf_counter()
            metrics.append({**timing,'at':done,'queue_ms':(process_at-arrival)*1000,
                'submit_ms':(done-send_at)*1000,'arrival_to_submit_ms':(done-arrival)*1000})
            if done>=poll_at:
                state=live_status();states.append(state);poll_at=done+1
                if not state['fluxPaused'] or state['epoch']!=before['epoch']:
                    raise RuntimeError('Show mode/scene changed; stopping isolated trial')
            if observer is None and done>=observe_at:
                observer=LatestReceiver('TULIP ('+name+')',1296,720,16)
            if observer:
                sample=observer.take()
                if sample is not None:
                    observed+=1
                    if observed in (1,16,64):Image.fromarray(sample[1]).save(args.out/f'ndi-output-{observed}.png')
                    sample_hashes.add(hash(sample[1][::32,::32].tobytes()))
        elapsed=time.perf_counter()-start
        if observed<16 or len(sample_hashes)<4:raise RuntimeError('Output NDI did not deliver changing frames')
        report={'kind':args.kind,'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'seconds':elapsed,'frames_processed':len(metrics),'processed_fps':len(metrics)/elapsed,
            'source_slot_drops':source.dropped,'output_frames_verified':observed,
            'different_output_samples':len(sample_hashes),'before':before,'during':states,
            'after':live_status(),'metrics':metrics,'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20}
        for key in ('pipeline_ms','queue_ms','submit_ms','arrival_to_submit_ms'):
            report[key]={p:float(np.percentile([m[key] for m in metrics],q)) for p,q in [('p50',50),('p95',95)]}
        (args.out/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k not in ('metrics','during','before','after')}),flush=True)
    finally:
        source.close()
        if observer:observer.close()
        if publisher:
            try:
                bridge_request(publisher,2,'video_sender_stop',{'senderId':sender})
            finally:publisher.close()


if __name__=='__main__':main()
