"""Explicit-device NVIDIA VSR Ultra with independently owned presentation frames."""
import sys
from pathlib import Path

import numpy as np
import torch


class VsrUpscaler:
    def __init__(self, device):
        self.device = torch.device(device)
        if self.device.type != 'cuda' or self.device.index is None:
            raise ValueError('VSR requires an explicitly indexed CUDA device')
        package = Path(__file__).resolve().parents[3] / 'tulip/research/vfx-probe'
        if not (package / 'nvvfx').is_dir():
            raise RuntimeError('Pinned nvidia-vfx installation is missing')
        sys.path.insert(0, str(package))
        import nvvfx
        from nvvfx.effects.video_super_res import QualityLevel
        if nvvfx.__version__ != '0.1.0.1':
            raise RuntimeError('VSR requires tested nvidia-vfx 0.1.0.1')
        self._effect_factory = lambda: nvvfx.VideoSuperRes(quality=QualityLevel.ULTRA, device=self.device.index)
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.Stream(device=self.device, priority=-1)
            self.effect = self._effect_factory()
        self.output_size = None
        self.loaded = False
        self.input_tensor = None

    @torch.inference_mode()
    def process_gpu(self, frame, output_size=(720, 1296)):
        from .gpu_video_output import GpuFrame
        from .native_config import NATIVE_SIZES
        owned_frame = frame if isinstance(frame, GpuFrame) else None
        frame = owned_frame.tensor if owned_frame is not None else frame
        tensor_input = isinstance(frame, torch.Tensor)
        if (frame.ndim != 3 or frame.shape[2] != 3 or
                (frame.shape[1], frame.shape[0]) not in NATIVE_SIZES or
                frame.dtype != (torch.uint8 if tensor_input else np.uint8)):
            raise ValueError('VSR requires a supported native-size RGB uint8 frame')
        if tensor_input and frame.device != self.device:
            raise ValueError('VSR tensor input must already be on its output GPU')
        with torch.cuda.device(self.device):
            if tuple(output_size) != self.output_size:
                self.stream.synchronize()
                self.effect.output_width = output_size[1]
                self.effect.output_height = output_size[0]
                self.output_size = tuple(output_size)
                self.loaded = False
            if not self.loaded:
                self.effect.load()
                self.loaded = True
        with torch.cuda.device(self.device):
            # Accept work produced on another CUDA stream without a CPU round trip.
            # GpuFrame also carries its explicit producer event below.
            if tensor_input:
                self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            if owned_frame is not None:
                owned_frame.wait()
            if tensor_input:
                image = frame[:, :, :3].permute(2, 0, 1).contiguous().float() / 255
            else:
                image = torch.from_numpy(frame.copy()).permute(2, 0, 1).contiguous().to(
                    self.device, dtype=torch.float32) / 255
            if self.input_tensor is None or self.input_tensor.shape != image.shape:
                self.input_tensor = torch.empty_like(image)
            self.input_tensor.copy_(image)
            self.stream.synchronize()
            effect = self.effect._effect
            effect.set_input_image(frame.shape[1], frame.shape[0])
            effect.set_cuda_stream(self.stream.cuda_stream)
            effect.transfer_input(self.input_tensor, self.stream.cuda_stream)
            effect.run(False)
            # Pinned SDK 1.2 uses internal work beyond the supplied stream.
            # Its convenience run() exports before that work completes, yielding
            # black/stale frames. Synchronize BEFORE get_output, then wait for
            # the default-stream export. This also waits on other output-GPU
            # work; measure prompt-transition latency before accepting the trial.
            torch.cuda.synchronize(self.device)
            pixels = torch.from_dlpack(effect.get_output(self.device.index))
            torch.cuda.default_stream(self.device).synchronize()
            # The SDK reuses its output buffer. Conversion allocates owned storage
            # on this same stream before the next SDK call can overwrite it.
            pixels = pixels.clamp(0, 1).mul(255).byte().permute(1, 2, 0).contiguous()
            return GpuFrame.record(pixels)

    @torch.inference_mode()
    def process(self, frame, output_size=(720, 1296)):
        result = self.process_gpu(frame, output_size)
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            return result.wait().cpu().numpy()

    def close(self):
        with torch.cuda.device(self.device):
            self.stream.synchronize()
            self.effect.close()
