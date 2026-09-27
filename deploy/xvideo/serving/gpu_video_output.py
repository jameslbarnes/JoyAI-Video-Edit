"""GPU-resident presentation and NVENC output; CPU receives compressed bytes only.

The optional 1080p NDI / 720p preview branches remain independent consumers.
Uses the same bounded queue, fractional pacing and moving shader handoff as NDI.
"""
import ctypes
import collections
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
import struct
import json
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

from .direct_ndi_output import DirectNdiPublisher

# PyNvVideoCodec's ABGR/bt709 conversion produces full-range YUV but does not
# write corresponding VUI flags. Tag it explicitly without re-encoding pixels.
H264_COLOR_METADATA = 'h264_metadata=video_full_range_flag=1:colour_primaries=1:transfer_characteristics=13:matrix_coefficients=1'


@dataclass
class GpuFrame:
    tensor: torch.Tensor  # immutable HWC RGB uint8 on the output GPU
    ready: object

    @property
    def shape(self): return self.tensor.shape

    @classmethod
    def record(cls, tensor):
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(tensor.device))
        return cls(tensor, event)

    def wait(self):
        torch.cuda.current_stream(self.tensor.device).wait_event(self.ready)
        return self.tensor


class CudaRgba:
    """NVENC ABGR is byte-ordered RGBA on little-endian Windows."""
    def __init__(self, tensor, stream):
        self.tensor, self.stream = tensor, stream

    @property
    def __cuda_array_interface__(self):
        t = self.tensor
        return {
            'shape': tuple(t.shape), 'strides': tuple(t.stride()),
            'typestr': '|u1', 'data': (t.data_ptr(), False), 'version': 3,
            'stream': self.stream.cuda_stream}

    def cuda(self): return self


def annexb_nalus(packet):
    return [n for n in re.split(b'\x00\x00(?:\x00)?\x01', packet) if n]


class FlvH264:
    """Timestamped, zero-B-frame AVC tags avoid raw-H264 demuxer buffering."""
    header = b'FLV\x01\x01\x00\x00\x00\x09\x00\x00\x00\x00'

    def __init__(self):
        self.sps = self.pps = None
        self.sequence = None

    @staticmethod
    def tag(payload, timestamp):
        timestamp = int(timestamp)
        if len(payload) >= 1 << 24 or not 0 <= timestamp < 1 << 32:
            raise ValueError('FLV tag exceeds limits')
        head = b'\x09' + len(payload).to_bytes(3,'big') + (timestamp & 0xffffff).to_bytes(3,'big') + bytes([timestamp >> 24]) + b'\x00\x00\x00'
        return head + payload + (11+len(payload)).to_bytes(4,'big')

    def packet(self, data, timestamp):
        nalus = annexb_nalus(data)
        for n in nalus:
            if n[0] & 31 == 7: self.sps = n
            elif n[0] & 31 == 8: self.pps = n
        if self.sps is None or self.pps is None:
            raise ValueError('Encoder did not provide SPS/PPS before video')
        output = b''
        sequence = (self.sps,self.pps)
        if sequence != self.sequence:
            s,p = sequence
            config = b'\x01'+s[1:4]+b'\xff\xe1'+len(s).to_bytes(2,'big')+s+b'\x01'+len(p).to_bytes(2,'big')+p
            output += self.tag(b'\x17\x00\x00\x00\x00'+config,timestamp)
            self.sequence = sequence
        video = [n for n in nalus if n[0] & 31 not in (7,8,9)]
        if video:
            keyframe = any(n[0] & 31 == 5 for n in video)
            payload = bytes([0x17 if keyframe else 0x27])+b'\x01\x00\x00\x00'
            payload += b''.join(len(n).to_bytes(4,'big')+n for n in video)
            output += self.tag(payload,timestamp)
        return output


class GpuVideoPublisher(DirectNdiPublisher):
    def __init__(self, device, ffmpeg, url, bitrate=15000000):
        super().__init__()
        if not str(url).startswith('rtmp://127.0.0.1:1935/'):
            raise ValueError('GPU publisher accepts only the local MediaMTX RTMP endpoint')
        if not Path(ffmpeg).is_file(): raise ValueError('GPU publisher ffmpeg not found')
        self.device = torch.device(device)
        self.ffmpeg, self.url, self.bitrate = str(ffmpeg), url, int(bitrate)
        self.process = self.nvenc = self.stream = None
        self.encode_ms = self.packet_bytes = self.packet_count = 0
        self.raw_download_bytes = 0
        self.restart_count = 0
        self.transport_events = collections.deque(maxlen=32)
        self.stage_ms = {}
        self.delivery_gaps = collections.deque(maxlen=32)
        self._last_delivery_clock = None
        self.encoded = collections.deque()
        self.encode_ahead = 8
        self._sending = False
        self._transport_started = self._transport_starved = False
        self.pacer = None
        self.pacer_queued = 0

    def status(self):
        base = super().status()
        with self.condition:
            base.update(raw_queued=len(self.frames), encoded_queued=len(self.encoded),
                        queued=len(self.frames)+len(self.encoded)+self.pacer_queued,
                        max_frames=self.max_frames+self.encode_ahead+12,
                        paced_queued=self.pacer_queued)
        return {**base, 'backend':'gpu-nvenc', 'device':str(self.device),
                'url':self.url, 'bitrate':self.bitrate, 'encode_ms':self.encode_ms,
                'packet_bytes':self.packet_bytes, 'packet_count':self.packet_count,
                'raw_download_bytes':self.raw_download_bytes,
                'mux_pid':self.process.pid if self.process else None,
                'restart_count':max(0,self.restart_count-1),
                'transport_events':list(self.transport_events),
                'stream_priority':self.stream.priority if self.stream else None,
                'stage_ms':dict(self.stage_ms), 'delivery_gaps':list(self.delivery_gaps)}

    def _validate_frames(self, frames):
        shape = frames[0].shape
        if tuple(shape) not in ((720,1296,3),(1080,1944,3),(1440,2592,3),(2160,3840,3)):
            raise ValueError('Unsupported GPU output geometry')
        if any(not isinstance(f,GpuFrame) or f.shape != shape or f.tensor.dtype != torch.uint8 or f.tensor.device != self.device for f in frames):
            raise ValueError('GPU output needs matching immutable RGB tensors on its device')
        return shape

    def _resize_source(self, source, key):
        # Only the small native shader input crosses into this output GPU.
        w,h,_ = key
        source = np.asarray(source,dtype=np.uint8)
        if source.shape != (360,648,3): raise ValueError('GPU shader bridge must remain native sized')
        tensor = torch.from_numpy(source.copy()).to(self.device)
        tensor = F.interpolate(tensor.permute(2,0,1)[None].float(),size=(h,w),mode='bilinear',align_corners=False)
        return GpuFrame.record(tensor[0].permute(1,2,0).round_().clamp_(0,255).byte().contiguous())

    def _blend(self, first, second, mix):
        a,b = first.wait(),second.wait()
        return GpuFrame.record(torch.lerp(a.float(),b.float(),mix).round_().clamp_(0,255).byte())

    def _open(self, key):
        import PyNvVideoCodec as nvc
        w,h,fps = key
        context = ctypes.c_void_p()
        if ctypes.WinDLL('nvcuda.dll').cuCtxGetCurrent(ctypes.byref(context)) != 0 or not context.value:
            raise RuntimeError('No active Torch CUDA context for NVENC')
        self.nvenc = nvc.CreateEncoder(w,h,'ABGR',False,
            gpu_id=self.device.index,cudacontext=context.value,cudastream=self.stream.cuda_stream,
            codec='h264',preset='P4',tuning_info='ultra_low_latency',rc='cbr',
            bitrate=self.bitrate,maxbitrate=self.bitrate,vbvbufsize=20000000,
            fps=round(fps),gop=round(fps),idrperiod=round(fps),bf=0,lookahead=0,
            colorspace='bt709',repeatspspps=1,max_res=[w,h],qmin=0)
        if self.nvenc.GetNumBFrames() != 0: raise RuntimeError('GPU publisher requires no B-frame reordering')
        self.rgba = torch.empty((h,w,4),device=self.device,dtype=torch.uint8)
        self.rgba[:,:,3] = 255
        self.cuda_frame = CudaRgba(self.rgba,self.stream)
        self.flv = FlvH264()
        self.output_index = 0
        # Only encoded bytes cross this pipe. FFmpeg remuxes; it never decodes.
        log_path = Path(__file__).resolve().parents[3]/'tulip/gpu-output-mux.log'
        self.log = log_path.open('ab',buffering=0)
        command = [self.ffmpeg,'-hide_banner','-loglevel','warning','-nostdin',
            '-probesize','32768','-analyzeduration','0','-f','flv','-i','pipe:0',
            '-map','0:v:0','-c:v','copy','-bsf:v',H264_COLOR_METADATA,'-an','-flush_packets','1',
            '-flvflags','no_duration_filesize','-f','flv',self.url]
        self.process = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,
            stderr=self.log,bufsize=0,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.pacer = subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('encoded_pacer.py')),str(fps)],
            stdin=subprocess.PIPE,stdout=self.process.stdin,stderr=subprocess.PIPE,bufsize=0,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.restart_count += 1
        self.transport_events.append({'at':time.time(),'event':'open',
            'geometry':list(key),'mux_pid':self.process.pid,'pacer_pid':self.pacer.pid})
        sent_base,underrun_base=self.sent,self.underruns
        pacer=self.pacer
        def read_status():
            try:
                for line in pacer.stderr:
                    state=json.loads(line)
                    with self.condition:
                        if self.pacer is not pacer:return
                        self.sent=sent_base+state['sent']
                        self.underruns=underrun_base+state['underruns']
                        self.pacer_queued=state['queued']
                        self.last_send_at=state['last_send_at'];self.last_send_ms=state['last_send_ms']
                        gap=state.get('last_gap')
                        if gap and (not self.delivery_gaps or self.delivery_gaps[-1]['at']!=gap['at']):self.delivery_gaps.append(gap)
            except Exception as exc:
                if self.pacer is pacer and not self.stopping:self.error=f'Pacer telemetry: {exc}'
        threading.Thread(target=read_status,name='joyai-pacer-status',daemon=True).start()
        self._write(self.flv.header)
        self.geometry = key

    def _write(self, data):
        view = memoryview(struct.pack('<I',len(data))+data)
        while view:
            count = self.pacer.stdin.write(view)
            if not count: raise BrokenPipeError('GPU encoded-video pipe closed')
            view = view[count:]

    def _encode(self, frame, fps):
        prepared_at = time.perf_counter()
        self.rgba[:,:,:3].copy_(frame.wait())
        # Synchronize only the output stream before handing its pointer to NVENC.
        # The input wrapper retains the buffer until Encode's copy completes.
        self.stream.synchronize()
        start = time.perf_counter()
        prepare_ms = (start-prepared_at)*1000
        packets = self.nvenc.Encode(self.cuda_frame)
        self.encode_ms = (time.perf_counter()-start)*1000
        sync_at = time.perf_counter()
        self.stream.synchronize()
        packet_at = time.perf_counter()
        encoded = []
        for packet in packets:
            data = packet['data']
            encoded.append(self.flv.packet(data,round(self.output_index*1000/fps)))
            self.output_index += 1
            self.packet_count += 1
            self.packet_bytes += len(data)
        self.stage_ms = {'prepare':prepare_ms,'nvenc':self.encode_ms,
                         'post_sync':(packet_at-sync_at)*1000,
                         'packetize':(time.perf_counter()-packet_at)*1000}
        return encoded

    def _release_encoder(self):
        if self.process is not None:
            self.transport_events.append({'at':time.time(),'event':'release',
                'geometry':list(self.geometry) if self.geometry else None,
                'mux_pid':self.process.pid,'mux_exit':self.process.poll(),
                'error':self.error,'stopping':self.stopping})
        pacer,self.pacer=getattr(self,'pacer',None),None
        if pacer:
            if pacer.poll() is None:pacer.terminate()
            try:pacer.wait(timeout=3)
            except subprocess.TimeoutExpired:pacer.kill();pacer.wait(timeout=3)
            if pacer.stdin:pacer.stdin.close()
            if pacer.stderr:pacer.stderr.close()
        self.pacer_queued=0
        # Terminating our mux unblocks a blocked pipe before joining the worker.
        process,self.process = self.process,None
        if process:
            if process.poll() is None: process.terminate()
            try: process.wait(timeout=3)
            except subprocess.TimeoutExpired: process.kill();process.wait(timeout=3)
            if process.stdin: process.stdin.close()
        self.nvenc = None
        self.rgba = self.cuda_frame = None
        if getattr(self,'log',None): self.log.close();self.log=None

    def close(self):
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        # Worker cleanup can clear either field after the stop notification.
        # Retain local handles through poll/terminate instead of rereading them.
        process, pacer = self.process, self.pacer
        if pacer and pacer.poll() is None:pacer.terminate()
        if process and process.poll() is None: process.terminate()
        super().close()

    def _send_encoded(self):
        """Feed the isolated clock ahead; OS pipe backpressure bounds storage."""
        try:
            while True:
                with self.condition:
                    while not self.stopping:
                        if not self.encoded:
                            self.condition.wait(.01);continue
                        frame,key,packets,stages,transition = self.encoded.popleft()
                        self._sending = True
                        self._transport_started = True
                        self._transport_starved = False
                        self.condition.notify_all()
                        break
                    else:return
                start=time.perf_counter()
                self._write(b''.join(packets))
                with self.condition:
                    self.last_frame,self.last_key=frame,key
                    self._sending=False
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                if not self.stopping:self.error=f'{type(exc).__name__}: {exc}'
                self.stopping=True
                self._sending=False
                self.condition.notify_all()

    @torch.inference_mode()
    def _run(self):
        deadline = time.perf_counter()
        key = None
        self.encoded.clear()
        self._transport_started=self._transport_starved=False
        sender=threading.Thread(target=self._send_encoded,name='joyai-encoded-pacer',daemon=True)
        sender.start()
        try:
            with torch.cuda.device(self.device):
                # Encode ahead on the GPU; the independent CPU sender owns the
                # display deadlines and absorbs bounded GPU scheduling stalls.
                self.stream = torch.cuda.Stream(device=self.device, priority=-1)
                with torch.cuda.stream(self.stream):
                    while True:
                        with self.condition:
                            while not self.stopping:
                                if len(self.encoded)>=self.encode_ahead:
                                    self.condition.wait(.02);continue
                                if not self.frames and self.transition is None:
                                    self.condition.wait(.1);continue
                                # Sample a moving shader at real time, not in
                                # an eight-frame burst of identical snapshots.
                                delay = deadline-time.perf_counter() if self.transition is not None else 0
                                if delay > 0: self.condition.wait(delay);continue
                                selected = self._next_frame(time.perf_counter())
                                if selected is None: self.condition.wait(.01);continue
                                frame,next_key = selected
                                break
                            else: return
                        if next_key != key:
                            with self.condition:
                                while (self.encoded or self._sending) and not self.stopping:
                                    self.condition.wait(.01)
                                if self.stopping:return
                            self._release_encoder()
                            self._open(next_key)
                            key = next_key
                        start = time.perf_counter()
                        packets=self._encode(frame,key[2])
                        with self.condition:
                            self.encoded.append((frame,key,packets,dict(self.stage_ms),self.transition is not None))
                            self.condition.notify_all()
                        # The shader is sampled in real time, but needs supply
                        # headroom just like generated output. An exact-rate
                        # producer accumulates scheduler misses and drains the
                        # isolated sender across overlapping transitions.
                        transition_fps = key[2] * (4/3 if self.pacer_queued < 8 else 1)
                        period = 1/(transition_fps if self.transition is not None else key[2])
                        deadline = deadline+period if self.transition is not None else start+period
                        if deadline < start: deadline = start+period
        except Exception as exc:
            with self.condition:
                if not self.stopping:
                    self.error = f'{type(exc).__name__}: {exc}'
                    self.dropped += len(self.frames)
                self.frames.clear()
        finally:
            with self.condition:
                self.stopping=True
                self.condition.notify_all()
            if self.process and self.process.poll() is None:self.process.terminate()
            if self.pacer and self.pacer.poll() is None:self.pacer.terminate()
            sender.join(timeout=3)
            self._release_encoder()
            with self.condition:
                self.encoded.clear()
