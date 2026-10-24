"""Local, fixed-preset upstream WebSocket service; no model reimplementation."""
# ruff: noqa: E402
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))
from xvideo.serving.windows_priority import configure_delivery_priority
print(json.dumps({"deliveryPriority": configure_delivery_priority()}), flush=True)
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=7873)
options = parser.parse_args()
preset = json.loads((ROOT / "tulip/presets/etherea-continuous-v2.json").read_text())
from xvideo.serving.native_config import load_native_config
native_quality = load_native_config()
preset['start'].update({k: native_quality[k] for k in ('width', 'height', 'fps')})

import torch
import uvicorn
from xvideo.serving.serve_joyomni_streaming import build_parser, create_app

physical = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
physical_free = float(subprocess.check_output([
    "nvidia-smi", "-i", physical, "--query-gpu=memory.free",
    "--format=csv,noheader,nounits",
], text=True).strip()) * 2**20
free, total = torch.cuda.mem_get_info()
budget = min(free, physical_free) - int(0.75 * 2**30)
if budget < 21 * 2**30:
    raise RuntimeError(f"JoyAI preset needs >=21 GiB allocator headroom; available {budget / 2**30:.2f} GiB. Other services were not changed.")
torch.cuda.set_per_process_memory_fraction(budget / total)
print(json.dumps({"preset": preset["id"], "physical_gpu": physical,
                  "allocator_budget_gib": budget / 2**30}), flush=True)
models = ROOT / "tulip/models"
args = build_parser().parse_args([
    "--dit-ckpt", str(models / "JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth"),
    "--vae-ckpt", str(models / "JoyAI-Video-Edit/vae"),
    "--text-encoder-ckpt", str(models / "MiMo-VL-7B-RL-2508"),
    "--device", "cuda:0", "--vae-device", os.environ.get('JOYAI_VAE_DEVICE', 'cuda:0'),
    "--vae-encode-device", os.environ.get('JOYAI_VAE_ENCODE_DEVICE', os.environ.get('JOYAI_VAE_DEVICE', 'cuda:0')),
    "--vae-pseudo-device", os.environ.get('JOYAI_VAE_PSEUDO_DEVICE', os.environ.get('JOYAI_VAE_DEVICE', 'cuda:0')),
    # Packing stays alongside decode: moving only the single-frame pseudo
    # encoder should not copy every decoded frame to the other GPU as float32.
    "--postprocess-device", os.environ.get('JOYAI_VAE_DEVICE', 'cuda:0'),
    "--height", str(native_quality['height']), "--width", str(native_quality['width']), "--fps", str(native_quality['fps']),
    "--num-inference-steps", "2", "--seed", "42",
    "--no-use-pe", "--no-online-gate", "--no-freeze-kv-on-static",
    "--kv-reset-frames", "0", "--max-temporal-ids", str(preset["start"]["max_temporal_ids"]),
    "--output-quality", "95", "--uplink-codec", "mjpeg",
    "--downlink-codec", "mjpeg", "--no-profile-timings", "--no-prebake-graph",
    "--prompt", preset["prompt"],
])
app = create_app(args)
from xvideo.serving.spout_input import SpoutInput, load_input_config
def create_spout_input():
    config = load_input_config()
    if config.get("transport") != "spout":
        raise ValueError("Spout input is not enabled")
    return SpoutInput(config["source"], config["device"])
app.state.spout_input_factory = create_spout_input
from xvideo.serving.shader_strength import ShaderStrength, load_strength_config
app.state.shader_strength = ShaderStrength(**load_strength_config(ROOT / 'tulip/shader-strength.json'),
                                           config_path=ROOT / 'tulip/shader-strength.json')
from xvideo.serving.cache_decay import CacheDecay, load_decay_config
app.state.cache_decay = CacheDecay(**load_decay_config(ROOT / 'tulip/cache-decay.json'),
                                   native_fps=native_quality['fps'])
from xvideo.serving.history_noise import HistoryNoise, load_history_noise_config
app.state.history_noise = HistoryNoise(**load_history_noise_config(ROOT / 'tulip/history-noise.json'))

@app.get('/native-config')
def native_config():
    return native_quality


class LocalPreset:
    """Reject cross-site browser callers and pin the tested shader configuration."""
    def __init__(self, application):
        self.application = application

    async def __call__(self, scope, receive, send):
        if scope["type"] in {"http", "websocket"}:
            headers = dict(scope.get("headers", []))
            origin = headers.get(b"origin")
            if origin and urlparse(origin.decode()).hostname not in {"127.0.0.1", "localhost", "tulip"}:
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                else:
                    await send({"type": "http.response.start", "status": 403, "headers": []})
                    await send({"type": "http.response.body", "body": b"Local callers only"})
                return

        async def pinned_receive():
            message = await receive()
            if message["type"] == "websocket.receive" and message.get("text"):
                try:
                    payload = json.loads(message["text"])
                except (ValueError, TypeError):
                    return message
                if isinstance(payload, dict) and payload.get("type") == "start":
                    spout_requested = payload.get("spout_input") is True
                    # Configuration changes take effect only at an explicit new
                    # session. The single-session service keeps weights resident.
                    updated = load_native_config()
                    native_quality.update(updated)
                    args.width, args.height, args.fps = updated['width'], updated['height'], updated['fps']
                    preset['start'].update({k: updated[k] for k in ('width', 'height', 'fps')})
                    payload = {**preset["start"], "prompt": str(payload.get("prompt") or preset["prompt"])[:12000],
                               "input_codec": "spout" if spout_requested and load_input_config().get("transport") == "spout" else "mjpeg",
                               "independent_preview": payload.get("independent_preview") is True,
                               "profile_timings": False,
                               "scene_token": payload.get("scene_token"),
                               "ref_image": payload.get("ref_image") if isinstance(payload.get("ref_image"), str) and len(payload["ref_image"]) <= 4 * 1024 * 1024 else None,
                               "rife_multiplier": native_quality['rife_multiplier'] if (ROOT / 'tulip/native-quality.json').exists()
                                   else payload["rife_multiplier"] if payload.get("rife_multiplier") in (2, 4) else 1}
                    message = {**message, "text": json.dumps(payload)}
            return message

        await self.application(scope, pinned_receive, send)


if __name__ == "__main__":
    uvicorn.run(LocalPreset(app), host="127.0.0.1", port=options.port,
                loop="asyncio", ws_max_size=2 * 1024 * 1024,
                ws_max_queue=2, ws_per_message_deflate=False)
