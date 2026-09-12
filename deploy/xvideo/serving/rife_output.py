"""Optional 2x presentation-only RIFE; never feeds interpolated pixels to KV/VAE.

Loads the user's existing FluxRT implementation/checkpoint read-only. Original
JPEGs are passed unchanged; only midpoint JPEGs are encoded. State is per session.
"""
import importlib.util
import io
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file


class RifeOutput:
    def __init__(self, device="cuda:0"):
        root = Path(os.environ["JOYAI_RIFE_FLUX_ROOT"]).resolve()
        source = root / "src/fluxrt/stream_processor/interpolation_model.py"
        spec = importlib.util.spec_from_file_location("joyai_output_ifnet", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.device = torch.device(device)
        module.device = self.device
        self.model = module.IFNet().eval()
        self.model.load_state_dict(load_file(str(root / "RIFE-safetensors/flownet.safetensors")))
        self.model.to(self.device, torch.float16)
        self.reset()

    def reset(self):
        self.previous = None
        self.previous_meta = None

    @torch.inference_mode()
    def process(self, frames, metas, profile, quality=95, *, raw_output=False):
        start = time.perf_counter()
        output, correspondence = [], []
        with torch.cuda.device(self.device):
            if not frames:
                return [], []
            if len(frames) > 8 or len(frames) != len(metas):
                raise ValueError("RIFE accepts at most one matched eight-frame chunk")
            arrays = np.stack([np.asarray(Image.open(io.BytesIO(j)).convert("RGB"))
                               if isinstance(j, bytes) else j for j in frames])
            batch = torch.from_numpy(arrays).permute(0, 3, 1, 2).to(self.device, torch.float16) / 255
            h, w = batch.shape[-2:]
            batch = F.pad(batch, (0, (-w) % 32, 0, (-h) % 32), mode="replicate")
            has_previous = self.previous is not None and self.previous.shape[1:] == batch.shape[1:]
            chain = torch.cat([self.previous, batch]) if has_previous else batch
            pixels = None
            if len(chain) > 1:
                middle = self.model(torch.cat([chain[:-1], chain[1:]], dim=1))
                pixels = middle[:, :, :h, :w].clamp(0, 1).mul(255).byte().permute(0, 2, 3, 1).cpu().numpy()
            for i, (jpeg, meta) in enumerate(zip(frames, metas)):
                if has_previous or i > 0:
                    midpoint = pixels[i if has_previous else i-1]
                    if raw_output:
                        output.append(midpoint)
                    else:
                        buffer = io.BytesIO()
                        Image.fromarray(midpoint).save(buffer, format="JPEG", quality=quality)
                        output.append(buffer.getvalue())
                    correspondence.append({**meta, "interpolated": True,
                        "source_seq_previous": (metas[i-1] if i else self.previous_meta).get("seq"), "interpolation_t": 0.5})
                output.append(arrays[i] if raw_output else jpeg)
                correspondence.append({**meta, "interpolated": False})
            self.previous, self.previous_meta = batch[-1:].clone(), metas[-1]
        # Midpoint .cpu() above synchronizes our work, not every JoyAI stream.
        profile["rife_s"] = time.perf_counter() - start
        profile["generated_count"] = len(frames)
        profile["display_count"] = len(output)
        profile["rife_multiplier"] = 2
        return output, correspondence
