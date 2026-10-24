"""Explicit-device NVIDIA VSR with independently owned presentation frames."""

import numpy as np
import torch

from .vfx_runtime import VsrSettings, load_bindings


class VsrUpscaler:
    def __init__(self, device, *, settings=None):
        self.device = torch.device(device)
        if self.device.type != 'cuda' or self.device.index is None:
            raise ValueError('VSR requires an explicitly indexed CUDA device')
        self.settings = settings if settings is not None else VsrSettings.from_startup()
        nvvfx, quality_enum = load_bindings(self.settings)
        self.sdk_version = nvvfx.get_sdk_version()
        self._effect_factory = lambda: nvvfx.VideoSuperRes(
            **self.settings.effect_arguments(self.device.index, quality_enum))
        with torch.cuda.device(self.device):
            # SDK 1.3 Streaming Ultra fails/returns black frames with the legacy
            # stream-switching path. Keep its input, effect and RGB8 export on
            # the device's default stream, as in NVIDIA's blocking API example.
            # Preserve the accepted SDK 1.2 path for explicit rollback.
            self.stream = (torch.cuda.default_stream(self.device)
                           if self.settings.release == '0.2.0.0'
                           else torch.cuda.Stream(device=self.device, priority=-1))
            self.effect = self._effect_factory()
        self.output_size = None
        self.input_size = None
        self.loaded = False
        self.input_tensor = None

    def status(self):
        return {'package_version': self.settings.release, 'sdk_version': self.sdk_version,
                'quality': int(self.effect.quality), 'quality_name': self.effect.quality.name,
                'strength': getattr(self.effect, 'strength', None),
                'strength_source': 'sdk_default' if self.settings.release == '0.1.0.1' else 'explicit',
                'device': str(self.device), 'loaded': self.loaded,
                'implementation': ('official_bindings_default_stream_sync'
                                   if self.settings.release == '0.2.0.0'
                                   else 'official_bindings_conservative_sync')}

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
            if self.settings.release == '0.2.0.0':
                input_size = (frame.shape[1], frame.shape[0])
                if input_size != self.input_size:
                    self.stream.synchronize()
                    self.input_size = input_size
                    self.loaded = False
                if not self.loaded:
                    # Streaming modes require input geometry before load;
                    # configuring it only at run produced NVVFX_ERR_INITIALIZATION.
                    self.effect._effect.set_input_image(*input_size)
                    self.effect._effect.set_cuda_stream(self.stream.cuda_stream)
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
            # The accepted SDK 1.2 path uses work beyond the supplied stream.
            # Its convenience run() exports before that work completes, yielding
            # black/stale frames. Synchronize BEFORE get_output, then wait for
            # the default-stream export. This also waits on other output-GPU
            # work; measure prompt-transition latency before accepting the trial.
            # Retain the conservative producer wait for both releases. SDK 1.3
            # additionally completes the RGB8 export before importing it below.
            torch.cuda.synchronize(self.device)
            capsule = effect.get_output(self.device.index)
            if self.settings.release == '0.2.0.0':
                # Complete the SDK's RGB8 export before importing/copying its
                # borrowed output. A configured-stream wait alone is insufficient.
                torch.cuda.synchronize(self.device)
            pixels = torch.from_dlpack(capsule)
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


# Compatibility name used by the isolated implementation-comparison test.
LegacyVsrUpscaler = VsrUpscaler
