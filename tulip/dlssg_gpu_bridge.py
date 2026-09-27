"""Isolated DLSSG bridge adapter; retain GPU surfaces through downstream consumers."""
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'tulip/research/visual-enhancer-v11/app'

def load_bridge():
    # Load only the interpolation modules, without the desktop/image application.
    # The caller selects isolated PyAV 18 before importing any video modules.
    for name,relative in [('src','src'),('src.core','src/core'),('src.frame_interpolation','src/frame_interpolation')]:
        if name not in sys.modules:
            module=types.ModuleType(name);module.__path__=[str(APP/relative)];sys.modules[name]=module
    from src.frame_interpolation import native
    from src.core.jobs import JobController
    from src.core.neural_bridge import _DLPackPlane
    return native,JobController,_DLPackPlane

class GpuDlssg:
    def __init__(self,width=1152,height=648,device=0,multiplier=2,*,torch_context=False,own_output=False):
        native,JobController,self.plane_type=load_bridge()
        if torch_context:
            # The RGB/Torch path never constructs FFmpeg AVFrames. Retain the
            # existing Torch CUDA primary context instead of requiring PyAV 18
            # merely to create the same device context through FFmpeg.
            def ensure_primary(ordinal):
                import torch
                with torch.cuda.device(ordinal):
                    torch.cuda.init()
                    torch.empty(1,device=f'cuda:{ordinal}')
            native._MANAGER._ensure_ffmpeg_device=ensure_primary
        self.native=native
        self.status=native.initialize_bridge(device)
        self.session=native.DirectDLSSGSession(width,height,multiplier-1,JobController(),device,timeout=30)
        self.status=native._MANAGER.status()
        self.device=device
        self.own_output=own_output

    def tensor(self,surface):
        import torch
        d=surface.descriptor
        if d.pixel_format!=self.native.FORMAT_RGBA8:
            surface.close();raise ValueError('Expected SDR uint8 RGBA GPU surface')
        plane=self.plane_type(pointer=d.planes[0],shape=(d.height,d.width,4),
            strides=(d.strides[0],4,1),bits=8,device_id=self.device,
            retain=surface.retain,release=surface.release)
        try:
            # Native processing completes before returning its exported CUDA surface.
            # DLPack retains that surface until the final Torch view is released.
            pixels=torch.from_dlpack(plane)
            assert pixels.device==torch.device('cuda',self.device)
            rgb=pixels[:,:,:3]
            if self.own_output:
                # Keep lookahead/output queues independent of the bridge's
                # bounded surface pool. This is a GPU-to-GPU copy, no readback.
                rgb=rgb.clone(memory_format=torch.contiguous_format)
                torch.cuda.current_stream(self.device).synchronize()
            return rgb
        finally:
            surface.close()

    def process(self,rgba,*,reset=False):
        import torch
        from xvideo.serving.gpu_video_output import GpuFrame
        with torch.cuda.device(self.device):
            generated,detail=self.session.process_frame(rgba,force_reset=reset,output_rgb=True,color_range=1)
            tensors=[]
            try:
                for surface in generated:tensors.append(self.tensor(surface))
                tensors.append(self.tensor(self.session.copy_current_input_rgb()))
                return [GpuFrame.record(t) for t in tensors],detail
            finally:
                for surface in generated:surface.close()

    def close(self):
        self.session.close()
