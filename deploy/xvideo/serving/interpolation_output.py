"""Explicit, chunk-boundary RIFE/DLSSG selection with GPU-owned DLSSG output."""
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[3]
SETTINGS=ROOT/'tulip/interpolation-settings.json'


def selected_backend(path=SETTINGS):
    settings=json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}
    backend=settings.get('backend','rife')
    if backend not in ('rife','dlssg'):
        raise ValueError('Interpolation backend must be rife or dlssg')
    return backend


class DlssgOutput:
    def __init__(self,device):
        self.device=torch.device(device)
        if self.device.type!='cuda' or self.device.index is None:
            raise ValueError('DLSSG requires an explicitly indexed CUDA device')
        sys.path.insert(0,str(ROOT/'tulip'))
        from dlssg_gpu_bridge import GpuDlssg
        self.factory=GpuDlssg
        self.models={}
        self.key=None
        with torch.cuda.device(self.device):
            self.stream=torch.cuda.Stream(device=self.device)
        self.reset()

    def reset(self,*,transition=False):
        self.previous=None
        self.previous_meta=None
        self.reset_pending=True
        self.transition_pending=False

    @torch.inference_mode()
    def process(self,frames,metas,profile,quality=95,*,raw_output=False,
                transition_source=None,multiplier=2,gpu_output=False):
        if multiplier not in (2,4):raise ValueError('DLSSG multiplier must be 2 or 4')
        if not raw_output or not gpu_output:
            raise ValueError('Live DLSSG requires the GPU VSR presentation path')
        if not frames:return [],[]
        if len(frames)>8 or len(frames)!=len(metas):
            raise ValueError('DLSSG accepts one matched chunk of at most eight frames')
        started=time.perf_counter();output=[];correspondence=[];details=[];holds=0
        with torch.cuda.device(self.device),torch.cuda.stream(self.stream):
            for frame,meta in zip(frames,metas):
                pixels=np.asarray(Image.open(io.BytesIO(frame)).convert('RGB')) if isinstance(frame,bytes) else np.asarray(frame)
                if pixels.dtype!=np.uint8 or pixels.ndim!=3 or pixels.shape[2]!=3:
                    raise ValueError('DLSSG expects native RGB uint8 frames')
                h,w=pixels.shape[:2];key=(w,h,multiplier)
                if key!=self.key:
                    self.reset();self.key=key
                if key not in self.models:
                    if len(self.models)>=4:raise RuntimeError('DLSSG geometry cache full; restart explicitly')
                    self.models[key]=self.factory(w,h,self.device.index,multiplier,
                                                  torch_context=True,own_output=True)
                model=self.models[key]
                rgba=np.empty((h,w,4),dtype=np.uint8);rgba[:,:,:3]=pixels;rgba[:,:,3]=255
                generated,detail=model.process(rgba,reset=self.reset_pending)
                self.reset_pending=False;details.append(detail)
                current=generated[-1];midpoints=generated[:-1]
                if self.previous is not None:
                    if len(midpoints)>multiplier-1:raise RuntimeError('Unexpected DLSSG frame count')
                    missing=multiplier-1-len(midpoints)
                    # At a detected cut/disabled startup interval, hold the
                    # previous frame for cadence. Never silently invoke RIFE.
                    for i,middle in enumerate([self.previous]*missing+midpoints):
                        output.append(middle)
                        correspondence.append({**meta,'interpolated':True,
                            'interpolation_backend':'dlssg','interpolation_t':(i+1)/multiplier,
                            'source_seq_previous':self.previous_meta.get('seq'),
                            'cadence_hold':i<missing})
                    holds+=missing
                output.append(current);correspondence.append({**meta,'interpolated':False})
                self.previous=current;self.previous_meta=dict(meta)
        elapsed=time.perf_counter()-started
        profile.update(interpolation_backend='dlssg',interpolation_s=elapsed,
            interpolation_device=str(self.device),interpolation_gpu_output=True,
            generated_count=len(frames),display_count=len(output),rife_multiplier=multiplier)
        profile['dlssg']={'runtime':model.status,'frames':len(frames),'cadence_holds':holds,
            'generated_midpoints':len(output)-len(frames)-holds,
            'upload_bytes':sum(d['upload_bytes'] for d in details),
            'download_bytes':sum(d['download_bytes'] for d in details),
            'native_ms':sum(d['total_ms'] for d in details),
            'scene_cuts':sum(d['scene_cut'] for d in details),
            'optical_flow_active':all(d['nvof_active'] for d in details)}
        return output,correspondence

    def close(self):
        with torch.cuda.device(self.device):
            self.stream.synchronize();self.previous=None
            for model in self.models.values():model.close()
        self.models.clear()


class InterpolationOutput:
    """Retain both engines for an explicit live switch; never auto-roll back."""
    def __init__(self,device):
        self.device=torch.device(device)
        self.backend=selected_backend()
        self.models={}
        self._transition=False
        from .native_capture import NativeCaptureTap
        self.capture=NativeCaptureTap()

    @property
    def supports_scene_morph(self):return self.backend=='rife'

    @property
    def transition_pending(self):
        return self._transition and self.supports_scene_morph

    def reset(self,*,transition=False):
        self._transition=bool(transition)
        self.capture.reset()
        for model in self.models.values():model.reset(transition=transition)

    def process(self,frames,metas,profile,*args,**kwargs):
        self.capture.observe(frames,metas,profile)
        backend=selected_backend()
        changed=backend!=self.backend
        if backend not in self.models:
            if backend=='rife':
                from .rife_output import RifeOutput
                model=RifeOutput(self.device)
            else:model=DlssgOutput(self.device)
            model.reset(transition=self._transition)
            self.models[backend]=model
        elif changed:self.models[backend].reset()
        self.backend=backend
        result=self.models[backend].process(frames,metas,profile,*args,**kwargs)
        profile['interpolation_backend']=backend
        self._transition=False
        return result

    def close(self):
        self.capture.close()
        for model in self.models.values():
            if hasattr(model,'close'):model.close()
        self.models.clear()
