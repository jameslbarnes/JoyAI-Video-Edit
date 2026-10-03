from __future__ import annotations

import os
import queue
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from PIL import Image

from xvideo.config import ExpConfig, generate_video_image_bucket
from xvideo.models.models import load_dit, load_pipeline, build_vae
from xvideo.models.pipeline import (
    PRECISION_TO_TYPE,
)
from xvideo.serving.graph_runner import GRAPH_WINDOW_CHUNKS, StreamingGraphRunner, graph_env_enabled, fixed_prompt_graph_enabled, prompt_graph_capacity, prompt_graph_capacities
from xvideo.utils import _dynamic_resize_from_bucket, seed_everything

DEFAULT_REFERENCE_IMG_IV2V_BASESIZE = 768

try:
    from torchvision.io import encode_jpeg as _tv_encode_jpeg
    _NVJPEG_OK = torch.cuda.is_available()
except Exception:  # noqa: BLE001
    _tv_encode_jpeg = None
    _NVJPEG_OK = False

def _autocast_ctx(device_type: str, dtype: torch.dtype, enabled: bool):
    if device_type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device_type, dtype=dtype, enabled=enabled)
    return nullcontext()


_FULL_WARMUP_CHUNKS = 4
_GRAPH_CACHE_CAP = 1
_GRAPH_CAPTURE_MAX_FAILS = 2

def _vae_compile_module():
    from xvideo.models.vae import vae_compile as module
    return module

@dataclass(frozen=True)
class _ProfileTimer:
    wall_start: float
    start_event: torch.cuda.Event | None = None

def _profile_timer_start(
    device: torch.device | str | None = None,
    *,
    use_cuda_event: bool = True,
) -> _ProfileTimer:
    if not use_cuda_event or device is None or not torch.cuda.is_available():
        return _ProfileTimer(wall_start=time.perf_counter())
    device_obj = torch.device(device)
    if device_obj.type != "cuda":
        return _ProfileTimer(wall_start=time.perf_counter())
    start_event = torch.cuda.Event(enable_timing=True)
    start_event.record(torch.cuda.current_stream(device_obj))
    return _ProfileTimer(wall_start=time.perf_counter(), start_event=start_event)

def _profile_timer_elapsed(started: _ProfileTimer | float) -> float:
    if isinstance(started, _ProfileTimer):
        if started.start_event is not None:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            end_event.synchronize()
            return float(started.start_event.elapsed_time(end_event)) / 1000.0
        return time.perf_counter() - started.wall_start
    return time.perf_counter() - float(started)

@dataclass(frozen=True)
class StreamingSettings:
    height: int = 720
    width: int = 1248
    num_inference_steps: int = 2
    seed: int = 42
    max_sequence_length: int | None = None
    enable_denormalization: bool | None = None
    max_temporal_ids: int | None = None
    freeze_kv_on_static: bool = True
    static_diff_thresh: float = 0.5
    store_clean_self_only: bool = True
    profile_timings: bool = False
    output_codec: str = "mjpeg"

@dataclass
class StreamingChunkResult:
    profile: dict[str, Any]
    source_metas: list[dict[str, Any]]
    elapsed: float
    jpegs: list[bytes] | None = None
    valid_count: int | None = None

@dataclass
class _ChunkJob:
    chunk_idx: int
    source_frames: list[Image.Image]
    source_metas: list[dict[str, Any]]
    profile: dict[str, Any]
    total_started: float
    frozen_anchor_id: int | None = None
    valid_count: int | None = None
    source_pixels: torch.Tensor | None = None  # original GPU upload, THWC uint8

@dataclass
class _EncodedChunk:
    job: _ChunkJob
    ref_chunk_latent: torch.Tensor
    ready_event: torch.cuda.Event | None = None

@dataclass
class _DenoisedChunk:
    job: _ChunkJob
    current_chunk_latents: torch.Tensor
    ready_event: torch.cuda.Event | None = None

@dataclass
class _DecodedPixelsChunk:
    job: _ChunkJob
    decoded_pixels: torch.Tensor
    ready_event: torch.cuda.Event | None = None

def _record_ready_event() -> torch.cuda.Event | None:
    if not torch.cuda.is_available():
        return None
    ev = torch.cuda.Event()
    ev.record()
    return ev

def _consume_on_current_stream(ev: torch.cuda.Event | None, *tensors: torch.Tensor) -> None:
    if ev is None:
        return
    stream = torch.cuda.current_stream()
    stream.wait_event(ev)
    for t in tensors:
        if t is not None and t.is_cuda:
            t.record_stream(stream)

def _module_device(module: torch.nn.Module) -> torch.device:
    if hasattr(module, "device"):
        return torch.device(module.device)
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")

def _load_vae_for_device(cfg: ExpConfig, device: torch.device) -> torch.nn.Module:
    vae = build_vae(cfg, device)
    vae = vae.to(device)
    vae.requires_grad_(False)
    vae.eval()
    return vae


def _canonical_device(device: torch.device | str) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _clone_vae_shared_weights(src: torch.nn.Module) -> torch.nn.Module:
    """A second VAE handle sharing `src`'s weight tensors.

    The VAE keeps per-call streaming state on the instance, so each role needs
    its own module object -- but weights can be shared. Conv weights are
    converted to channels_last_3d up-front: vae_compile converts per instance,
    and Tensor.to(memory_format=...) returns self once the layout already
    matches, so every clone keeps sharing storage.
    """
    from xvideo.models.vae import XVAEChunkCausal

    for m in src.modules():
        if isinstance(m, torch.nn.Conv3d):
            m.weight.data = m.weight.data.to(memory_format=torch.channels_last_3d)
    with torch.device("meta"):
        clone = XVAEChunkCausal.from_config(src.config)
    clone.load_state_dict(src.state_dict(), assign=True)
    clone.requires_grad_(False)
    return clone.eval()

class JoyOmniRuntime:
    def __init__(
        self,
        cfg: ExpConfig,
        pipeline: Any,
        device: torch.device,
        *,
        decode_vae: torch.nn.Module | None = None,
        pseudo_encode_vae: torch.nn.Module | None = None,
        postprocess_device: torch.device | None = None,
    ):
        self.cfg = cfg
        self.pipeline = pipeline
        self.device = device
        self.decode_vae = decode_vae or pipeline.vae
        self.pseudo_encode_vae = pseudo_encode_vae or self.decode_vae
        self.postprocess_device = postprocess_device or _module_device(self.pseudo_encode_vae)
        self.dit_lock = threading.RLock()
        self.text_lock = threading.Lock()
        self.presentation_lock = threading.RLock()
        self.interpolation_lock = threading.RLock()
        from xvideo.serving.stage_timings import StageTimings
        self.stage_timings = StageTimings()
        # The runtime outlives sessions. Keep VAE allocator ownership on the
        # same streams when a session is replaced at unchanged resolution.
        self.vae_worker_streams = {}
        self.vae_submission_locks = {}
        self.output_quality = 60
        self.lossless_output = False
        self.graph_runners: dict[tuple, StreamingGraphRunner] = {}
        self.graph_capture_failures: dict[tuple, int] = {}

    def run_presentation(self, function, *args, **kwargs):
        # Graph capture must exclude RIFE/SR work on every device in this process.
        with self.presentation_lock:
            return function(*args, **kwargs)

    def run_interpolation(self, function, *args, **kwargs):
        with self.interpolation_lock:
            return function(*args, **kwargs)

    @classmethod
    def load(
        cls,
        dit_ckpt: str | None,
        *,
        vae_ckpt: str | None = None,
        text_encoder_ckpt: str | None = None,
        device: str | torch.device | None = None,
        vae_device: str | torch.device | None = None,
        vae_encode_device: str | torch.device | None = None,
        vae_decode_device: str | torch.device | None = None,
        vae_pseudo_device: str | torch.device | None = None,
        postprocess_device: str | torch.device | None = None,
        seed: int = 42,
        warmup_height: int = 720,
        warmup_width: int = 1248,
    ) -> "JoyOmniRuntime":
        device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        encode_device_arg = vae_encode_device if vae_encode_device is not None else vae_device
        decode_device_arg = vae_decode_device if vae_decode_device is not None else vae_device
        pseudo_device_arg = vae_pseudo_device if vae_pseudo_device is not None else decode_device_arg
        vae_encode_device_obj = torch.device(encode_device_arg) if encode_device_arg is not None else None
        vae_decode_device_obj = torch.device(decode_device_arg) if decode_device_arg is not None else None
        vae_pseudo_device_obj = torch.device(pseudo_device_arg) if pseudo_device_arg is not None else None
        postprocess_device_obj = torch.device(postprocess_device) if postprocess_device is not None else vae_pseudo_device_obj
        seed_everything(seed)

        from xvideo.lowvram import log_mode
        log_mode()

        cfg = ExpConfig()
        if vae_ckpt is not None:
            cfg.vae_arch_config["pretrained"] = vae_ckpt
        if text_encoder_ckpt is not None:
            cfg.text_encoder_arch_config["params"]["text_encoder_ckpt"] = text_encoder_ckpt
        if dit_ckpt is not None:
            cfg.dit_ckpt = dit_ckpt


        dit = load_dit(cfg, device=device_obj)
        if getattr(dit.config, "causal", False):
            dit.config.use_inference_kv_cache = True
        dit.requires_grad_(False)
        dit.eval()

        from xvideo.models.dit import attention_backend
        print(f"#####[STREAM] attention backend: {attention_backend()}", flush=True)

        pipeline = load_pipeline(cfg, dit, device_obj)
        pipeline.vae.requires_grad_(False)
        pipeline.vae.eval()
        if vae_encode_device_obj is not None:
            pipeline.vae = pipeline.vae.to(vae_encode_device_obj)
            print(f"#####[STREAM] moved encode VAE to {vae_encode_device_obj}")

        def _vae_for_role(role: str, target_device: torch.device) -> torch.nn.Module:
            target = _canonical_device(target_device)
            for share_src in (pipeline.vae, decode_vae):
                if share_src is not None and _canonical_device(_module_device(share_src)) == target:
                    print(f"#####[STREAM] {role} VAE shares weights on {target_device}")
                    return _clone_vae_shared_weights(share_src)
            vae = _load_vae_for_device(cfg, target_device)
            print(f"#####[STREAM] loaded {role} VAE on {target_device}")
            return vae

        decode_vae = None
        if vae_decode_device_obj is not None:
            decode_vae = _vae_for_role("decode", vae_decode_device_obj)
        if decode_vae is None:
            decode_vae = pipeline.vae
        pseudo_encode_vae = decode_vae
        if vae_pseudo_device_obj is not None:
            pseudo_encode_vae = _vae_for_role("pseudo encode", vae_pseudo_device_obj)

        if hasattr(pipeline, "set_progress_bar_config"):
            pipeline.set_progress_bar_config(disable=True)
        pipeline.transformer.eval()

        _orientations = [(warmup_height, warmup_width)]
        native_warmup_only = os.environ.get("JOYOMNI_WARMUP_NATIVE_ONLY", "0") == "1"
        if not native_warmup_only and (warmup_width, warmup_height) != (warmup_height, warmup_width):
            _orientations.append((warmup_width, warmup_height))
        _stem_mod = pipeline.vae.stem

        try:
            _vc = _vae_compile_module()
            _lat_c = int(getattr(decode_vae, "latent_channels", 0) or 0)
            _fspa = int(getattr(decode_vae, "ffactor_spatial", 0) or 0)
            _ddev = _module_device(decode_vae)
            _ddt = PRECISION_TO_TYPE[cfg.vae_precision]
            if _lat_c > 0 and _fspa > 0:
                for (_wh, _ww) in _orientations:
                    _ch = _wh * _stem_mod.group // _stem_mod.stride
                    _cw = _ww * _stem_mod.group // _stem_mod.stride
                    _vc.warmup_decode(
                        decode_vae, _lat_c,
                        _ch // _fspa, _cw // _fspa,
                        device=_ddev, dtype=_ddt,
                        # Graph scheduler accumulates/returns FP32 latents even
                        # with BF16 DiT/VAE compute. Warm the actual input guard.
                        input_dtype=(torch.float32 if graph_env_enabled() else
                                     PRECISION_TO_TYPE[cfg.dit_precision]),
                    )
        except Exception as _vc_exc:
            print(f"#####[STREAM] VAE compile warmup skipped: {_vc_exc!r}")

        try:
            _vc = _vae_compile_module()
            _vae_dt = PRECISION_TO_TYPE[cfg.vae_precision]
            _vae_ac = (_vae_dt != torch.float32)

            _src_vae = pipeline.vae
            for (_wh, _ww) in _orientations:
                _vc.warmup_encode(
                    _src_vae, 3, _wh, _ww,
                    device=_module_device(_src_vae), dtype=_vae_dt,
                    temporal_lens=(1, 1 + int(getattr(_src_vae, "ffactor_temporal", 8) or 8)),
                    autocast=_vae_ac,
                )
                _vc.warmup_encode(
                    pseudo_encode_vae, 3, _wh, _ww,
                    device=_module_device(pseudo_encode_vae), dtype=_vae_dt,
                    temporal_lens=(1,),
                    autocast=_vae_ac,
                )

            _ref_basesize = getattr(cfg, "ref_image_basesize", DEFAULT_REFERENCE_IMG_IV2V_BASESIZE)
            _ref_cfgs = generate_video_image_bucket(
                img_basesize=_ref_basesize, bs_img=1, bs_vid=0, bs_mimg=0, bs_mvid=0,
            )
            _ref_hw = sorted({(c[3], c[4]) for c in _ref_cfgs})
            _vc.maybe_setup_encode_dynamic(_src_vae)
            _vc.warmup_encode_dynamic(
                _src_vae, 3, [] if native_warmup_only else _ref_hw,
                device=_module_device(_src_vae), dtype=_vae_dt,
                temporal_lens=(1,),
                autocast=False,
            )
        except Exception as _ve_exc:
            print(f"#####[STREAM] VAE encode compile warmup skipped: {_ve_exc!r}")

        runtime = cls(
            cfg=cfg,
            pipeline=pipeline,
            device=device_obj,
            decode_vae=decode_vae,
            pseudo_encode_vae=pseudo_encode_vae,
            postprocess_device=postprocess_device_obj,
        )

        try:
            if os.environ.get("JOYOMNI_SKIP_LOAD_WARMUP", "0").lower() in {"1", "true", "yes", "on"}:
                print("#####[STREAM] load-time warmup skipped (JOYOMNI_SKIP_LOAD_WARMUP); "
                      "warm up later on a real GPU")
            else:
                for (_wh, _ww) in _orientations:
                    runtime.warmup_full_pipeline(height=_wh, width=_ww)
        except Exception as _wexc:
            print(f"#####[STREAM] full-pipeline warmup error (non-fatal): {_wexc!r}")

        if device_obj.type == "cuda":
            _free_b, _total_b = torch.cuda.mem_get_info(device_obj)
            print(
                f"#####[STREAM] runtime loaded; device memory used "
                f"{(_total_b - _free_b) / 2**30:.1f}/{_total_b / 2**30:.1f} GiB "
                f"(torch reserved {torch.cuda.memory_reserved(device_obj) / 2**30:.1f} GiB, "
                f"allocated {torch.cuda.memory_allocated(device_obj) / 2**30:.1f} GiB)",
                flush=True,
            )

        return runtime

    def create_v2v_session(
        self,
        prompt: str,
        *,
        settings: StreamingSettings | None = None,
        ref_image: Image.Image | None = None,
    ) -> "JoyOmniV2VStreamingSession":
        return JoyOmniV2VStreamingSession(
            runtime=self,
            prompt=prompt,
            settings=settings or StreamingSettings(),
            ref_image=ref_image,
        )

    def prebake_graph(
        self,
        prompt: str,
        *,
        settings: StreamingSettings,
        ref_image: Image.Image | None = None,
    ) -> None:
        session = self.create_v2v_session(prompt, settings=settings, ref_image=ref_image)
        arr = np.zeros((settings.height, settings.width, 3), dtype=np.uint8)
        try:
            session._initialize(Image.fromarray(arr, mode="RGB"))
        finally:
            session.close()

    def warmup_full_pipeline(
        self,
        *,
        height: int = 720,
        width: int = 1248,
        num_chunks: int = _FULL_WARMUP_CHUNKS,
    ) -> None:
        t0 = time.time()

        ffactor_t = int(self.pipeline.vae_scale_factor_temporal)
        n_frames = 1 + max(0, num_chunks - 1) * ffactor_t + ffactor_t
        print(f"#####[STREAM] full-pipeline warmup: {num_chunks} chunks (~{n_frames} frames) at {width}x{height}")
        session = None
        try:
            settings = StreamingSettings(
                height=height,
                width=width,
                max_temporal_ids=8,
                profile_timings=False,
            )
            session = self.create_v2v_session(
                prompt="warmup", settings=settings,
            )
            rng = np.random.default_rng(0)
            completed = 0
            for i in range(n_frames):
                arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
                frame = Image.fromarray(arr, mode="RGB")
                results = session.push_frame(frame, frame_meta={"seq": i + 1, "t_capture_ms": 0.0})
                completed += len(results)
                if completed >= num_chunks:
                    break

            deadline = time.time() + 120.0
            while completed < num_chunks and time.time() < deadline:
                r = session.wait_async_result(timeout=0.5)
                if r is not None:
                    completed += 1
            print(f"#####[STREAM] full-pipeline warmup done: {completed} chunks in {time.time() - t0:.1f}s")
        except Exception as exc:
            print(f"#####[STREAM] full-pipeline warmup skipped/failed: {exc!r}")
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception:
                    pass

class JoyOmniV2VStreamingSession:
    _session_serial_counter = 0
    def __init__(
        self,
        *,
        runtime: JoyOmniRuntime,
        prompt: str,
        settings: StreamingSettings,
        ref_image: Image.Image | None = None,
    ):
        self.runtime = runtime
        JoyOmniV2VStreamingSession._session_serial_counter += 1
        self._session_serial = JoyOmniV2VStreamingSession._session_serial_counter
        self.pipeline = runtime.pipeline

        self.decode_vae = runtime.decode_vae
        self.pseudo_encode_vae = runtime.pseudo_encode_vae
        self.postprocess_device = runtime.postprocess_device
        self.cfg = runtime.cfg
        self.raw_prompt = prompt
        self.prompt = prompt
        self.settings = settings
        self.ref_image = ref_image.convert("RGB") if ref_image is not None else None
        self.ref_image_latent: torch.Tensor | None = None
        self.device = self.pipeline.transformer.device
        self.generator = torch.Generator(device=self.device).manual_seed(settings.seed)

        self._prev_static_gray: np.ndarray | None = None
        self._static_anchor_id: int | None = None

        self.target_dtype = PRECISION_TO_TYPE[self.cfg.dit_precision]
        self.vae_dtype = PRECISION_TO_TYPE[self.cfg.vae_precision]
        self.vae_autocast_enabled = self.vae_dtype != torch.float32
        self.autocast_enabled = self.target_dtype != torch.float32
        self.device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device).split(":", 1)[0]

        self.chunk_size = self.pipeline._resolve_streaming_chunk_size(None, 1)
        if self.chunk_size != 1:
            raise ValueError(
                "The online v2v session currently supports transformer chunk_size=1 only; "
                f"got {self.chunk_size}."
            )

        self.ffactor_t = int(self.pipeline.vae_scale_factor_temporal)
        self.latent_channels = int(self.pipeline.vae.config.latent_channels)
        _stem = self.pipeline.vae.stem
        assert settings.height % (_stem.stride * 8) == 0 and settings.width % (_stem.stride * 8) == 0, (
            f"session size must be a multiple of {_stem.stride * 8}, got {settings.width}x{settings.height}"
        )
        self.latent_h = settings.height * _stem.group // _stem.stride // int(self.pipeline.vae_scale_factor)
        self.latent_w = settings.width * _stem.group // _stem.stride // int(self.pipeline.vae_scale_factor)
        self.local_window_size = int(getattr(self.pipeline.transformer.config, "local_window_size", 1))
        self.global_sink_chunk = self.pipeline._resolve_global_sink_chunk(None, self.pipeline.transformer)
        self.enable_denormalization = (
            self.cfg.enable_denormalization
            if settings.enable_denormalization is None
            else settings.enable_denormalization
        )

        self._submit_lock = threading.RLock()
        self.initialized = False
        self.chunk_idx = 0
        self.pending_frames: list[Image.Image] = []
        self.pending_metas: list[dict[str, Any]] = []
        self.prev_source_frame: torch.Tensor | None = None
        self.ref_image_kv_prefilled = False

        self.streaming_cond_embeds: torch.Tensor | None = None
        self.last_chunk_profile: dict[str, Any] | None = None
        self._prompt_lock = threading.Lock()
        self._prompt_thread = None
        self._pending_prompt = self._prepared_prompt = None
        self._prompt_request = self._prompt_revision = 0
        self._prompt_status = "idle"
        self._prompt_error = None
        self._prompt_encode_s = 0.0
        self._latest_source_frame = None
        self._closing = False
        self._sink_refresh_chunks = max(0, int(os.environ.get("JOYOMNI_SINK_REFRESH_CHUNKS", "0")))
        self._sink_chunk = self._sink_refresh_count = 0
        self._sink_decay_chunks = 0
        self._sink_decay_owned = None
        from .cache_decay import CacheDecay
        self.cache_decay = CacheDecay()
        from .history_noise import HistoryNoise
        self.history_noise = HistoryNoise()
        self._history_generator = None
        self._cache_stats = {}
        from .shader_strength import ShaderStrength
        self.shader_strength = ShaderStrength()

        self._encode_queue: queue.Queue[_ChunkJob | None] | None = None
        self._dit_queue: queue.Queue[_EncodedChunk | None] | None = None
        self._decode_queue: queue.Queue[_DenoisedChunk | None] | None = None

        self._pseudo_queue: queue.Queue[_DecodedPixelsChunk | None] | None = None
        self._postprocess_queue: queue.Queue[_DecodedPixelsChunk | None] | None = None
        self._result_queue: queue.Queue[StreamingChunkResult] | None = None
        self._workers: list[threading.Thread] = []
        self._worker_error: BaseException | None = None
        self._worker_error_lock = threading.Lock()
        self._pseudo_latent_condition = threading.Condition()
        self._pseudo_latent_chunk_idx: int | None = None
        self._pseudo_latent: torch.Tensor | None = None
        self._pseudo_latent_ready_event: torch.cuda.Event | None = None
        self._encode_stream: torch.cuda.Stream | None = None
        self._decode_stream: torch.cuda.Stream | None = None
        self._pseudo_stream: torch.cuda.Stream | None = None
        self._post_stream: torch.cuda.Stream | None = None
        self._debug_lock = threading.Lock()
        self._debug_started_at = time.time()
        self._debug_worker_states: dict[str, dict[str, Any]] = {}
        self._debug_counters: dict[str, int] = {
            "submitted_chunks": 0,
            "encoded_chunks": 0,
            "denoised_chunks": 0,
            "decoded_pixel_chunks": 0,
            "pseudo_encoded_chunks": 0,
            "postprocessed_chunks": 0,
            "returned_results": 0,
        }

    def _timer_start(
        self,
        device: torch.device | str | None = None,
        *,
        use_cuda_event: bool = True,
    ) -> _ProfileTimer | float:
        if self.settings.profile_timings:
            return _profile_timer_start(device, use_cuda_event=use_cuda_event)
        return time.perf_counter()

    def _timer_record(
        self,
        profile: dict[str, float | int],
        key: str,
        started: float,
    ) -> None:
        if not self.settings.profile_timings:
            return
        profile[key] = float(profile.get(key, 0.0)) + _profile_timer_elapsed(started)

    def _record_queue_wait(
        self,
        profile: dict[str, Any],
        key: str,
        started: float,
    ) -> None:
        if not self.settings.profile_timings:
            return
        profile[key] = time.perf_counter() - started

    @property
    def frames_per_next_chunk(self) -> int:
        return 1 if self.chunk_idx == 0 else self.ffactor_t

    def _clear_vae_feature_caches(self) -> None:
        seen: set[int] = set()
        for vae in (self.pipeline.vae, self.decode_vae, self.pseudo_encode_vae):
            if vae is None or id(vae) in seen:
                continue
            seen.add(id(vae))
            clear_cache = getattr(vae, "clear_cache", None)
            if callable(clear_cache):
                clear_cache()

    @torch.no_grad()
    def push_frame(
        self,
        frame: Image.Image,
        frame_meta: dict[str, Any] | None = None,
        *,
        drain_results: bool = True,
    ) -> list[StreamingChunkResult]:
        with self._submit_lock:
            self._raise_worker_error_if_needed()
            frame = self._resize_frame(frame)
            self._latest_source_frame = frame
            meta = frame_meta or {"seq": self.chunk_idx + len(self.pending_frames) + 1, "t_capture_ms": time.time() * 1000.0}
            if not self.initialized:
                self._initialize(frame)
            self.pending_frames.append(frame)
            self.pending_metas.append(meta)
            results: list[StreamingChunkResult] = []
            while len(self.pending_frames) >= self.frames_per_next_chunk:
                n = self.frames_per_next_chunk
                chunk_frames = self.pending_frames[:n]
                chunk_metas = self.pending_metas[:n]
                del self.pending_frames[:n]
                del self.pending_metas[:n]
                results.extend(self._submit_or_process_chunk(chunk_frames, chunk_metas))
            if drain_results:
                results.extend(self._drain_async_results())
            self._raise_worker_error_if_needed()
            return results

    @torch.no_grad()
    def flush_pending(self) -> None:
        if not self.initialized or not self.pending_frames:
            return
        valid = len(self.pending_frames)
        pad = self.ffactor_t - valid
        chunk_frames = self.pending_frames + [self.pending_frames[-1]] * pad
        chunk_metas = self.pending_metas + [self.pending_metas[-1]] * pad
        self.pending_frames = []
        self.pending_metas = []
        self._submit_or_process_chunk(chunk_frames, chunk_metas, valid_count=valid)
        self._raise_worker_error_if_needed()

    def close(self) -> None:
        with self._prompt_lock:
            self._closing = True
            self._pending_prompt = self._prepared_prompt = None
            prompt_thread = self._prompt_thread
        if prompt_thread is not None:
            prompt_thread.join()
        # Drain workers before invalidating tensors they may still be using.
        self._stop_async_workers()
        # Worker joins finish CPU submissions, not outstanding CUDA kernels.
        # A prepared next scene can immediately reuse the runtime's VAE/graph
        # storage, so finish all session devices before clearing shared caches.
        if torch.cuda.is_available():
            devices = {str(self.device), str(self.postprocess_device)}
            devices.update(str(_module_device(vae)) for vae in (
                self.pipeline.vae, self.decode_vae, self.pseudo_encode_vae,
            ) if vae is not None)
            for device in sorted(devices):
                if torch.device(device).type == "cuda":
                    torch.cuda.synchronize(device)
        self._clear_vae_feature_caches()
        self.pipeline.transformer.reset_inference_kv_cache()
        # No empty_cache: keep the freed KV/graph blocks cached in the allocator
        # so the next session's pools reuse them instead of fresh cudaMallocs.

    @torch.no_grad()
    def _initialize(self, first_frame: Image.Image) -> None:
        if self.initialized:
            return

        if not getattr(self.pipeline.transformer.config, "causal", False):
            raise ValueError("Online streaming requires a causal transformer config.")
        self.pipeline.transformer.config.use_inference_kv_cache = True

        self.ref_image_latent = self._encode_ref_image_latent()

        self._encode_streaming_prompt(first_frame)

        self._clear_vae_feature_caches()
        self.pipeline.transformer.reset_inference_kv_cache()
        if self.ref_image_latent is not None:
            self.pipeline._prefill_static_reference_kv_cache(
                self.pipeline.transformer,
                prompt_embeds=self.streaming_cond_embeds,
                reference_image_latents=self.ref_image_latent,
                transformer_dtype=self.target_dtype,
            )
            self.ref_image_kv_prefilled = True

        self._prepare_graph_family()

        self.initialized = True

    @torch.no_grad()
    def _encode_streaming_prompt(self, anchor_frame: Image.Image) -> None:
        prepared = getattr(self, "_initial_prompt_embeds", None)
        self.streaming_cond_embeds = (prepared if prepared is not None else
                                     self._encode_prompt_value(self.prompt, anchor_frame)).to(self.device)
        self._initial_prompt_embeds = None

    @torch.no_grad()
    def _encode_prompt_value(self, text: str, anchor_frame: Image.Image) -> torch.Tensor:
        prompt_image = self._resize_frame(anchor_frame)
        prompt = f"<|im_start|>user\n<image>\n{text}<|im_end|>\n"

        max_sequence_length = self.settings.max_sequence_length or int(self.cfg.text_token_max_length)
        text_device = getattr(self.pipeline, "text_execution_device", self.device)
        device_context = torch.cuda.device(text_device) if torch.device(text_device).type == "cuda" else nullcontext()
        with self.runtime.text_lock, device_context:
            prompt_embeds, _ = self.pipeline.encode_prompt(
                prompt=[prompt], images=[prompt_image], device=text_device,
                num_videos_per_prompt=1, max_sequence_length=max_sequence_length,
                template_type="video",
            )
            # CPU handoff completes encoder work before the DiT worker sees it.
            return prompt_embeds.detach().to("cpu")

    def queue_prompt(self, prompt: str) -> int:
        prompt = prompt.strip()
        if not prompt or len(prompt) > 12000:
            raise ValueError("prompt must contain 1..12000 characters")
        if not self.initialized or self._latest_source_frame is None:
            raise ValueError("wait for the first generated frame before updating the prompt")
        with self._prompt_lock:
            if self._closing:
                raise ValueError("session is closing")
            self._prompt_request += 1
            self._pending_prompt = (self._prompt_request, prompt, self._latest_source_frame.copy())
            self._prepared_prompt = None
            self._prompt_error = None
            self._prompt_status = "encoding"
            if self._prompt_thread is None:
                self._prompt_thread = threading.Thread(target=self._prepare_prompt_worker,
                                                       name="joyomni-prompt", daemon=True)
                self._prompt_thread.start()
            return self._prompt_request

    def _prepare_prompt_worker(self):
        while True:
            with self._prompt_lock:
                request = self._pending_prompt
                self._pending_prompt = None
                if request is None or self._closing:
                    self._prompt_thread = None
                    return
            revision, prompt, anchor = request
            started = time.perf_counter()
            try:
                embeds = self._encode_prompt_value(prompt, anchor)
                self._prepare_prompt_graph(embeds, revision)
                with self._prompt_lock:
                    if revision == self._prompt_request and not self._closing:
                        self._prepared_prompt = (revision, prompt, embeds)
                        self._prompt_encode_s = time.perf_counter() - started
                        self._prompt_status = "prepared"
            except Exception as exc:
                with self._prompt_lock:
                    if revision == self._prompt_request:
                        self._prompt_status = "error"
                        self._prompt_error = str(exc)
                print(f"#####[PROMPT] encoding failed; old prompt retained: {exc!r}", flush=True)

    @torch.no_grad()
    def _prepare_prompt_graph(self, embeds, revision):
        """Capture a changed text shape after draining work, preserving all KV state."""
        if not graph_env_enabled():
            return
        key = self._graph_key(embeds)
        if self.runtime.graph_runners.get(key) is not None:
            return
        if fixed_prompt_graph_enabled():
            # Live updates must never drain the pipeline for a graph capture.
            # A complete family is required before initialized becomes true.
            raise RuntimeError('Prompt graph was not prewarmed; active prompt retained')
        started = time.perf_counter()
        with self._submit_lock:
            deadline = time.monotonic() + 10.0
            while True:
                with self._prompt_lock:
                    if self._closing or revision != self._prompt_request:
                        return
                self._raise_worker_error_if_needed()
                with self._debug_lock:
                    counters = dict(self._debug_counters)
                submitted = counters.get("submitted_chunks", 0)
                if all(counters.get(stage, 0) >= submitted for stage in
                       ("postprocessed_chunks", "pseudo_encoded_chunks")):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("Timed out draining GPU stages for prompt graph capture")
                time.sleep(0.005)
            # No new model jobs enter; all existing GPU stages have submitted
            # their final work. Output processing has its own worker-thread lock.
            with self.runtime.dit_lock, self.runtime.presentation_lock, self.runtime.interpolation_lock:
                previous = self.streaming_cond_embeds
                try:
                    for index in range(torch.cuda.device_count()):
                        torch.cuda.synchronize(index)
                    self.streaming_cond_embeds = embeds.to(device=self.device, dtype=self.target_dtype)
                    self._maybe_prepare_graph_runner()
                finally:
                    self.streaming_cond_embeds = previous
        self._prompt_graph_prepare_s = time.perf_counter() - started
        print(f"#####[PROMPT] graph prepared revision={revision} "
              f"seconds={self._prompt_graph_prepare_s:.3f} cache_preserved=True", flush=True)

    def _apply_prepared_prompt(self, chunk_idx, profile):
        with self._prompt_lock:
            prepared = self._prepared_prompt
            self._prepared_prompt = None
            if prepared is None or prepared[0] != self._prompt_request:
                return
        revision, prompt, embeds = prepared
        started = time.perf_counter()
        self.streaming_cond_embeds = embeds.to(device=self.device, dtype=self.target_dtype)
        self.prompt = self.raw_prompt = prompt
        # The encoder worker prepares a missing shape while GPU stages are
        # drained. Only bind here, at the next chunk boundary; never capture
        # concurrently with VAE work or alter the prompt's token sequence.
        runner = self.runtime.graph_runners.get(self._graph_key())
        profile["prompt_graph_reused"] = bool(runner is not None)
        profile["prompt_graph_capture_deferred"] = bool(runner is None)
        if runner is not None:
            runner.bind_session(self._session_serial, self.streaming_cond_embeds)
        with self._prompt_lock:
            self._prompt_revision = revision
            if revision == self._prompt_request:
                self._prompt_status = "applied"
        profile["prompt_apply_s"] = time.perf_counter() - started
        print(f"#####[PROMPT] revision={revision} chunk={chunk_idx} encode={self._prompt_encode_s:.3f}s "
              f"apply={profile['prompt_apply_s']:.3f}s cache_preserved=True", flush=True)

    def _history_rng(self):
        """Dedicated generator: history noise never shifts the denoise noise sequence."""
        if self._history_generator is None:
            self._history_generator = torch.Generator(device=self.device).manual_seed(
                int(getattr(self.settings, "seed", 0) or 0) + 7919)
        return self._history_generator

    def _decay_sink(self, store, chunk_idx):
        """Fade the scene's first-chunk sink toward the latest committed chunk (see cache_decay)."""
        from .cache_decay import chunk_alpha, decay_sink, own_sink, rope_compatible
        config = self.cache_decay.snapshot()
        every = config["every_chunks"]
        chunk_s = self.frames_per_next_chunk / config["native_fps"]
        alpha = chunk_alpha(config["half_life_s"], chunk_s * every)
        state = {**config, "alpha": alpha, "chunks": self._sink_decay_chunks, "skipped": None}
        if alpha <= 0 or chunk_idx < 1 or chunk_idx % every:
            return state
        sink_id = self.pipeline._kv_cache_memory_id("clean", 0)
        sink, current = store.get(sink_id), store.get(self.pipeline._kv_cache_memory_id("clean", chunk_idx))
        if sink is None or current is None or sink is current:
            state["skipped"] = "missing"
            return state
        if not rope_compatible(sink, current):
            state["skipped"] = "rope"
            return state
        if sink is not self._sink_decay_owned:
            # The scene's original sink (or a refresh) may alias shared buffers.
            sink = store[sink_id] = self._sink_decay_owned = own_sink(sink)
        decay_sink(sink, current, alpha, config["mode"])
        runner = self.runtime.graph_runners.get(self._graph_key())
        if runner is not None:
            runner.pool_dirty = True  # graph pool holds a RoPE'd copy of the sink
        self._sink_decay_chunks += 1
        state["chunks"] = self._sink_decay_chunks
        return state

    def _refresh_sink_and_measure(self, chunk_idx, profile):
        store = self.pipeline.transformer._inference_kv_cache.get("cond", {})
        interval = self._sink_refresh_chunks
        if interval and chunk_idx > 1 and chunk_idx - self._sink_chunk >= interval:
            current = store[self.pipeline._kv_cache_memory_id("clean", chunk_idx)]
            # Independent storage is essential: graph publish buffers alternate
            # and will overwrite a sink that merely aliases the current entry.
            sink = {li: {**entry, "key": entry["key"].clone(), "value": entry["value"].clone()}
                    for li, entry in current.items()}
            store[self.pipeline._kv_cache_memory_id("clean", 0)] = sink
            runner = self.runtime.graph_runners.get(self._graph_key())
            if runner is not None:
                runner.pool_dirty = True  # re-RoPE the refreshed sink at position 0
            self._sink_chunk = chunk_idx
            self._sink_refresh_count += 1
        decay = self._decay_sink(store, chunk_idx)
        stats = {"entries": len(store), "sinkChunk": self._sink_chunk,
                 "sinkRefreshes": self._sink_refresh_count, "sinkRefreshChunks": interval,
                 "sinkDecay": decay,
                 "temporalPosition": profile.get("temporal_position"),
                 "allocatedMiB": torch.cuda.memory_allocated(self.device) / 2**20,
                 "reservedMiB": torch.cuda.memory_reserved(self.device) / 2**20}
        self._cache_stats = stats
        profile.update(prompt_revision=self._prompt_revision, prompt_status=self._prompt_status,
                       prompt_error=self._prompt_error, cache=stats)

    @torch.no_grad()
    def _encode_ref_image_latent(self) -> torch.Tensor | None:
        if self.ref_image is None:
            return None

        ref_image_basesize = getattr(self.cfg, "ref_image_basesize", DEFAULT_REFERENCE_IMG_IV2V_BASESIZE)
        ref_image_bucket_configs = generate_video_image_bucket(
            img_basesize=ref_image_basesize,
            bs_img=1,
            bs_vid=0,
            bs_mimg=0,
            bs_mvid=0,
        )
        resized_image, _ = _dynamic_resize_from_bucket(
            self.ref_image,
            bucket_configs=ref_image_bucket_configs,
            num_frames=1,
            num_items=1,
            return_bucket=True,
        )
        pixel = torch.from_numpy(np.array(resized_image))
        pixel = rearrange(pixel, "h w c -> c h w")
        if pixel.dtype != torch.uint8:
            pixel = pixel.clamp(0, 255).to(torch.uint8)
        pixel_normalized = pixel.to(torch.float32) / 127.5 - 1.0

        ref_img_tensor = rearrange(pixel_normalized, "c h w -> 1 c 1 h w")
        encode_device = _module_device(self.pipeline.vae)
        ref_img_encoded = ref_img_tensor.to(device=encode_device, dtype=self.vae_dtype)

        _vc = _vae_compile_module()
        encoded = _vc.encode_via_dynamic(self.pipeline.vae, ref_img_encoded)
        if not hasattr(encoded, "latent_dist"):
            raise TypeError(f"Unsupported VAE encode output type for ref image: {type(encoded)}")
        ref_img_latent = encoded.latent_dist.sample()
        if self.enable_denormalization:
            ref_img_latent = self.pipeline.normalize_latents(ref_img_latent)

        return ref_img_latent[:, :, :1].to(device=self.device, dtype=self.target_dtype)

    @staticmethod
    def _chunk_last_frame_gray(source_frames: list[Image.Image]) -> np.ndarray | None:
        if not source_frames:
            return None
        frame = source_frames[-1].convert("L").resize((64, 36))
        return np.asarray(frame, dtype=np.float32)

    def _update_static_anchor(self, chunk_idx: int, source_frames: list[Image.Image]) -> int | None:
        gray = self._chunk_last_frame_gray(source_frames)
        if not self.settings.freeze_kv_on_static or chunk_idx == 0 or gray is None:
            self._prev_static_gray = gray if gray is not None else self._prev_static_gray
            self._static_anchor_id = None
            return None

        anchor_id: int | None = None
        if self._prev_static_gray is not None:
            mad = float(np.mean(np.abs(gray - self._prev_static_gray)))
            if mad < self.settings.static_diff_thresh:
                if self._static_anchor_id is None:
                    self._static_anchor_id = chunk_idx - 1
                    print(
                        f"#####[FREEZE-KV] static detected at chunk_idx={chunk_idx} "
                        f"(mad={mad:.3f} < {self.settings.static_diff_thresh}), "
                        f"anchor={self._static_anchor_id}",
                        flush=True,
                    )
                anchor_id = self._static_anchor_id
            else:
                if self._static_anchor_id is not None:
                    print(
                        f"#####[FREEZE-KV] motion resumed at chunk_idx={chunk_idx} "
                        f"(mad={mad:.3f}), anchor released",
                        flush=True,
                    )
                self._static_anchor_id = None
                anchor_id = None

        self._prev_static_gray = gray
        return anchor_id

    def _submit_or_process_chunk(
        self,
        source_frames: list[Image.Image],
        source_metas: list[dict[str, Any]],
        valid_count: int | None = None,
    ) -> list[StreamingChunkResult]:
        self._submit_async_chunk(source_frames, source_metas, valid_count)
        return []

    def _new_profile(self, chunk_idx: int, input_frames: int) -> dict[str, Any]:
        return {
            "chunk_idx": chunk_idx,
            "input_frames": input_frames,
            "steps": int(self.settings.num_inference_steps),
            "profile_timings": int(self.settings.profile_timings),
            "dit_device": str(self.device),
            "vae_encode_device": str(_module_device(self.pipeline.vae)),
            "vae_decode_device": str(_module_device(self.decode_vae)),
            "vae_pseudo_device": str(_module_device(self.pseudo_encode_vae)),
            "postprocess_device": str(self.postprocess_device),
        }

    def _graph_runner_for_chunk(
        self,
        *,
        chunk_idx: int,
        selected_chunk_ids: list[int],
        gather_chunk_ids: list[int],
    ):
        mti = self.settings.max_temporal_ids
        if not graph_env_enabled() or mti is None or self.settings.num_inference_steps < 1:
            return None
        if self.chunk_size != 1 or len(selected_chunk_ids) != GRAPH_WINDOW_CHUNKS:
            return None
        if not self.settings.store_clean_self_only:
            return None
        if gather_chunk_ids != [0, chunk_idx - 1, chunk_idx]:
            return None
        if selected_chunk_ids[0] != 0 or selected_chunk_ids[-1] != chunk_idx:
            return None

        runner = self.runtime.graph_runners.get(self._graph_key())
        if runner is None or not getattr(runner, "ready", False):
            return None

        if runner.bound_token != self._session_serial:
            runner.bind_session(self._session_serial, self.streaming_cond_embeds)
        if runner.pool_dirty or runner.last_graph_chunk != chunk_idx - 1 or selected_chunk_ids[1] != chunk_idx - 1:
            scope_store = self.pipeline.transformer._inference_kv_cache["cond"]
            sink_mem_id = self.pipeline._kv_cache_memory_id("clean", 0)
            prev1_mem_id = self.pipeline._kv_cache_memory_id("clean", selected_chunk_ids[1])
            ref_mem_id = (
                self.pipeline._kv_cache_memory_id("ref_image")
                if self.ref_image_kv_prefilled else None
            )
            needed = [sink_mem_id, prev1_mem_id] + ([ref_mem_id] if ref_mem_id is not None else [])
            if any(mem_id not in scope_store for mem_id in needed):
                # History not in the python cache (session teardown drain / transition):
                # this chunk is simply not eligible; anything else raising below is a real bug.
                return None
            runner.seed_from_cache(
                scope_store,
                sink_mem_id=sink_mem_id,
                prev1_mem_id=prev1_mem_id,
                prev1_pos=min(chunk_idx - 1, int(mti) - 1),
                ref_mem_id=ref_mem_id,
            )
        runner.set_positions(chunk_idx)
        return runner

    def _ref_kv_tokens(self) -> int:
        if not self.ref_image_kv_prefilled:
            return 0
        scope_store = self.pipeline.transformer._inference_kv_cache["cond"]
        store = scope_store[self.pipeline._kv_cache_memory_id("ref_image")]
        return int(store[0]["key"].shape[1])

    def _graph_key(self, prompt_embeds=None) -> tuple:
        embeds = self.streaming_cond_embeds if prompt_embeds is None else prompt_embeds
        capacity = prompt_graph_capacity(int(embeds.shape[1]), int(self.settings.max_sequence_length or self.cfg.text_token_max_length)) if fixed_prompt_graph_enabled() else int(embeds.shape[1])
        if not 0 < int(embeds.shape[1]) <= capacity:
            raise ValueError("Prompt embeddings exceed the graph prompt capacity")
        return (
            self.latent_h,
            self.latent_w,
            capacity,
            self._ref_kv_tokens(),
            int(self.settings.max_temporal_ids or 0),
            int(self.settings.num_inference_steps),
            fixed_prompt_graph_enabled(),
        )

    def _prepare_graph_family(self) -> None:
        if not fixed_prompt_graph_enabled():
            self._maybe_prepare_graph_runner()
            return
        if not graph_env_enabled():
            return
        limit = int(self.settings.max_sequence_length or self.cfg.text_token_max_length)
        base_key = self._graph_key()
        for capacity in prompt_graph_capacities(limit):
            self._maybe_prepare_graph_runner(prompt_capacity=capacity)
            key = (*base_key[:2], capacity, *base_key[3:])
            runner = self.runtime.graph_runners.get(key)
            if runner is None or not runner.ready:
                raise RuntimeError(f'Could not prewarm prompt capacity {capacity}; playback not started')

    def _maybe_prepare_graph_runner(self, prompt_capacity=None) -> None:
        mti = self.settings.max_temporal_ids
        if not graph_env_enabled() or mti is None or self.chunk_size != 1:
            return
        if not self.settings.store_clean_self_only:
            return
        if not self.global_sink_chunk or self.local_window_size != GRAPH_WINDOW_CHUNKS:
            return
        key = self._graph_key()
        if prompt_capacity is not None:
            key = (*key[:2], prompt_capacity, *key[3:])
        cache = self.runtime.graph_runners
        fails = self.runtime.graph_capture_failures
        with self.runtime.dit_lock:
            existing = cache.get(key)
            if existing is not None and getattr(existing, "ready", False):
                return
            if fails.get(key, 0) >= _GRAPH_CAPTURE_MAX_FAILS:
                return
            torch.cuda.synchronize(self.device)
            if fixed_prompt_graph_enabled():
                family = (*key[:2], *key[3:])
                for old_key in list(cache):
                    if (*old_key[:2], *old_key[3:]) != family:
                        del cache[old_key]
                shared_kv = next(iter(cache.values()), None)
            else:
                shared_kv = None
                live_keys = [k for k, v in cache.items() if v is not None]
                while len(live_keys) >= _GRAPH_CACHE_CAP:
                    del cache[live_keys.pop(0)]
            cache.pop(key, None)
            # No empty_cache: the evicted runner's freed blocks are reused for the
            # new static bufs; returning them to the driver only slows recapture.
            pos_table_ids = torch.arange(int(mti), device=self.device, dtype=torch.long)
            latent_shape = (1, self.latent_channels, self.chunk_size, self.latent_h, self.latent_w)
            runner = StreamingGraphRunner(
                self.pipeline.transformer,
                chunk_tokens=self.latent_h * self.latent_w,
                ref_tokens=self._ref_kv_tokens(),
                latent_shape=latent_shape,
                txt_len=key[2],
                mask_prompt=fixed_prompt_graph_enabled(),
                shared_kv=shared_kv,
                max_temporal_ids=int(mti),
                pos_freqs=self._graph_cached_freqs(pos_table_ids),
                device=self.device,
                dtype=self.target_dtype,
                autocast_ctx=lambda: _autocast_ctx(
                    self.device_type, self.target_dtype, self.autocast_enabled
                ),
            )
            try:
                self.pipeline.scheduler.set_timesteps(
                    self.settings.num_inference_steps, device=self.device
                )
                runner.capture(
                    timesteps=self.pipeline.scheduler.timesteps,
                    sigmas=self.pipeline.scheduler.sigmas,
                )
            except Exception as exc:  # noqa: BLE001
                fails[key] = fails.get(key, 0) + 1
                print(
                    f"#####[GRAPH] capture failed (attempt {fails[key]}/{_GRAPH_CAPTURE_MAX_FAILS}), "
                    f"session stays eager: {exc!r}",
                    flush=True,
                )
                del runner
                torch.cuda.empty_cache()
                return
            cache[key] = runner

    def _graph_cached_freqs(self, cached_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tf = self.pipeline.transformer
        cached_frame_ids = tf._get_token_frame_ids(
            (int(cached_ids.numel()), self.latent_h, self.latent_w),
            self.device,
            temporal_ids=cached_ids,
        )
        cos, sin = tf.get_rotary_pos_embed_from_ids(
            frame_ids=cached_frame_ids,
            spatial_shape=(self.latent_h, self.latent_w),
        )
        return cos.unsqueeze(0), sin.unsqueeze(0)

    @torch.no_grad()
    def _denoise_chunk_graph(
        self,
        runner,
        ref_chunk_latent: torch.Tensor,
        current_chunk_latents: torch.Tensor,
        *,
        profile: dict[str, Any],
        history_chunk_ids: list[int],
        active_chunk_id: int,
    ) -> torch.Tensor:
        profile["graph_path"] = 1
        profile["graph_prompt_capacity"] = runner.txt_len
        profile["graph_prompt_tokens"] = runner.prompt_tokens
        profile["graph_prompt_masked"] = runner.mask_prompt
        profile["graph_trim_padded_text"] = hasattr(runner.prompt_key_mask, '_fp8_text_rows')
        profile["graph_active_text_rows"] = getattr(runner, 'active_text_rows', runner.txt_len)
        profile["graph_prompt_capacities"] = sorted(k[2] for k in self.runtime.graph_runners)
        profile["graph_prompt_attention"] = ("sage-prefix" if hasattr(runner.prompt_key_mask, '_sage_prefix') else "cudnn" if runner.mask_prompt else "unmasked")
        runner.in_ref_latent.copy_(ref_chunk_latent.to(self.target_dtype))
        sigma = self.history_noise.sigma
        profile["history_sigma"] = sigma
        if sigma > 0:
            runner.in_store_noise.copy_(torch.randn(runner.in_store_noise.shape, generator=self._history_rng(),
                                                    device=runner.in_store_noise.device, dtype=torch.float32))
        runner.in_store_sigma.fill_(sigma)
        runner.in_store_timestep.fill_(sigma * 1000.0)
        started = self._timer_start(self.device)
        with self.runtime.stage_timings.measure('dit_graph', self.device, profile.get('chunk_idx'),
                                               prompt_capacity=profile.get('graph_prompt_capacity')):
            runner.in_noise.copy_(current_chunk_latents.to(self.target_dtype))
            runner.full_graph.replay()
        self._timer_record(profile, "dit_forward_cond_s", started)

        keep_before_store = {self.pipeline._kv_cache_memory_id("clean", cid) for cid in history_chunk_ids}
        if self.ref_image_kv_prefilled:
            keep_before_store.add(self.pipeline._kv_cache_memory_id("ref_image"))
        self.pipeline.transformer.evict_kv_cache_chunks(keep_before_store)

        started = self._timer_start(self.device)
        scope_store = self.pipeline.transformer._inference_kv_cache["cond"]
        runner.publish_to_cache(
            scope_store, self.pipeline._kv_cache_memory_id("clean", active_chunk_id)
        )
        runner.last_graph_chunk = active_chunk_id
        self._timer_record(profile, "kv_store_forward_s", started)
        return runner.out_latents.clone()

    def _denoise_chunk(
        self,
        ref_chunk_latent: torch.Tensor,
        *,
        profile: dict[str, Any],
        chunk_idx: int,
        frozen_anchor_id: int | None = None,
    ) -> torch.Tensor:
        noise_shape = (1, self.latent_channels, self.chunk_size, self.latent_h, self.latent_w)
        current_chunk_latents = randn_tensor(
            noise_shape,
            generator=self.generator,
            device=self.device,
            dtype=self.target_dtype,
        )

        strength = self.shader_strength.advance()
        profile['shader_strength'] = strength
        from .shader_strength import prepare_strength_inputs
        current_chunk_latents, ref_chunk_latent = prepare_strength_inputs(
            current_chunk_latents, ref_chunk_latent, strength)

        total_latent_frames = chunk_idx + 1
        chunk_window = self.pipeline._get_last_chunk_window(
            total_latent_frames=total_latent_frames,
            chunk_size=self.chunk_size,
            window_size=self.local_window_size,
            global_sink_chunk=self.global_sink_chunk,
        )
        selected_chunk_ids = chunk_window["selected_chunk_ids"]
        history_chunk_ids = selected_chunk_ids[:-1]
        active_chunk_id = selected_chunk_ids[-1]

        gather_chunk_ids = selected_chunk_ids
        if (
            frozen_anchor_id is not None and
            history_chunk_ids and
            history_chunk_ids[-1] != frozen_anchor_id
        ):
            old_tail = history_chunk_ids[-1]

            fz_history = history_chunk_ids[:-1]
            anchor_is_dup = frozen_anchor_id in fz_history
            fz_history_with_anchor = fz_history if anchor_is_dup else fz_history + [frozen_anchor_id]
            fz_selected = fz_history_with_anchor + [active_chunk_id]

            disguised_tail = active_chunk_id - 1
            fz_gather_head = fz_history_with_anchor[:-1]
            degenerate = (
                anchor_is_dup or
                disguised_tail < 0 or
                disguised_tail in fz_gather_head
            )
            if not degenerate:
                history_chunk_ids = fz_history_with_anchor
                selected_chunk_ids = fz_selected
                gather_chunk_ids = fz_gather_head + [disguised_tail, active_chunk_id]
                print(
                    f"#####[FREEZE-KV] chunk_idx={chunk_idx} tail {old_tail}->anchor "
                    f"{frozen_anchor_id} (KV), pos-disguise->{disguised_tail}, "
                    f"kv_window={selected_chunk_ids} pos_window={gather_chunk_ids}",
                    flush=True,
                )
            else:
                print(
                    f"#####[FREEZE-KV] chunk_idx={chunk_idx} anchor={frozen_anchor_id} "
                    f"degenerate (dup_or_collision), falling back to normal path",
                    flush=True,
                )
        window_ids = self.pipeline._gather_window_temporal_ids(
            gather_chunk_ids,
            self.chunk_size,
            total_latent_frames,
            torch.device("cpu"),
            max_temporal_ids=self.settings.max_temporal_ids,
        )
        current_chunk_temporal_ids = window_ids[-self.chunk_size:]
        profile["temporal_position"] = int(current_chunk_temporal_ids[-1])
        cached_temporal_ids = window_ids[: -self.chunk_size]
        if cached_temporal_ids.numel() == 0:
            cached_temporal_ids = None

        runner = self._graph_runner_for_chunk(
            chunk_idx=chunk_idx,
            selected_chunk_ids=selected_chunk_ids,
            gather_chunk_ids=gather_chunk_ids,
        )
        if runner is not None:
            return self._denoise_chunk_graph(
                runner,
                ref_chunk_latent,
                current_chunk_latents,
                profile=profile,
                history_chunk_ids=history_chunk_ids,
                active_chunk_id=active_chunk_id,
            )

        self.pipeline.scheduler.set_timesteps(self.settings.num_inference_steps, device=self.device)
        timesteps_for_chunk = self.pipeline.scheduler.timesteps

        for timestep in timesteps_for_chunk:
            autocast_context = _autocast_ctx(
                self.device_type, self.target_dtype, self.autocast_enabled
            )
            with autocast_context:
                latent_model_input = current_chunk_latents.to(self.target_dtype)
                cache_memory_ids = [
                    self.pipeline._kv_cache_memory_id("clean", cid) for cid in history_chunk_ids
                ]
                if self.ref_image_kv_prefilled:
                    cache_memory_ids.append(self.pipeline._kv_cache_memory_id("ref_image"))

                t_expand = timestep.repeat(latent_model_input.shape[0])
                with self.pipeline.transformer.cache_context("cond"):
                    started = self._timer_start(self.device)
                    noise_pred = self.pipeline.transformer(
                        hidden_states=latent_model_input,
                        timestep=t_expand,
                        encoder_hidden_states=self.streaming_cond_embeds,
                        ref_video_latent=ref_chunk_latent,
                        current_temporal_ids=current_chunk_temporal_ids.unsqueeze(0).expand(
                            latent_model_input.shape[0], -1
                        ),
                        cached_temporal_ids=(
                            cached_temporal_ids.unsqueeze(0).expand(latent_model_input.shape[0], -1)
                            if cached_temporal_ids is not None
                            else None
                        ),
                        kv_cache_mode="reuse",
                        kv_cache_scope="cond",
                        kv_cache_chunk_id=active_chunk_id,
                        kv_cache_selected_chunk_ids=cache_memory_ids,
                        kv_cache_pre_rope=True,
                    )[0]
                    self._timer_record(profile, "dit_forward_cond_s", started)

                sample_for_step = current_chunk_latents.clone()
                current_chunk_latents = self.pipeline.scheduler.step(
                    noise_pred,
                    timestep,
                    sample_for_step,
                    return_dict=False,
                )[0]

        keep_before_store = {self.pipeline._kv_cache_memory_id("clean", cid) for cid in history_chunk_ids}
        if self.ref_image_kv_prefilled:
            keep_before_store.add(self.pipeline._kv_cache_memory_id("ref_image"))
        self.pipeline.transformer.evict_kv_cache_chunks(keep_before_store)

        started = self._timer_start(self.device)
        store_self_only = self.settings.store_clean_self_only
        store_history_chunk_ids = [] if store_self_only else history_chunk_ids
        store_mode = "store" if store_self_only else "reuse_store"
        store_cached_temporal_ids = None if store_self_only else cached_temporal_ids
        self._timer_record(profile, "kv_store_setup_s", started)

        started = self._timer_start(self.device)
        from .history_noise import noisy_history_latents
        # The scene-start sink is the identity anchor: store it clean. Noising it
        # washed the picture out toward white within minutes (sigma 0.20 replay).
        sigma = 0.0 if active_chunk_id == 0 else self.history_noise.sigma
        profile["history_sigma"] = sigma
        store_latents, store_timestep = noisy_history_latents(
            current_chunk_latents.to(self.target_dtype), sigma, self._history_rng() if sigma > 0 else None)
        self.pipeline._store_clean_chunk_kv_cache(
            self.pipeline.transformer,
            clean_chunk_latents=store_latents,
            timestep_value=store_timestep,
            chunk_temporal_ids=current_chunk_temporal_ids.unsqueeze(0).expand(
                current_chunk_latents.shape[0], -1
            ),
            prompt_embeds=self.streaming_cond_embeds,
            active_chunk_id=active_chunk_id,
            history_chunk_ids=store_history_chunk_ids,
            pre_rope=True,
            cached_temporal_ids=store_cached_temporal_ids,
            store_mode=store_mode,
        )
        self._timer_record(profile, "kv_store_forward_s", started)
        return current_chunk_latents

    def _evict_after_store(
        self, chunk_idx: int, frozen_anchor_id: int | None = None
    ) -> None:
        next_selected = self._next_selected_chunk_ids(chunk_idx)
        keep_after_store = {self.pipeline._kv_cache_memory_id("clean", cid) for cid in next_selected}
        if self.ref_image_kv_prefilled:
            keep_after_store.add(self.pipeline._kv_cache_memory_id("ref_image"))

        if frozen_anchor_id is not None:
            keep_after_store.add(self.pipeline._kv_cache_memory_id("clean", frozen_anchor_id))
        self.pipeline.transformer.evict_kv_cache_chunks(keep_after_store)

    def _finish_chunk_profile(
        self,
        profile: dict[str, Any],
        total_started: float,
    ) -> None:
        if self.settings.profile_timings:
            self._timer_record(profile, "total_server_chunk_s", total_started)
        else:
            profile["total_server_chunk_s"] = time.perf_counter() - total_started
        if self.settings.profile_timings:
            profile["dit_denoise_s"] = float(profile.get("dit_forward_cond_s", 0.0))
            profile["kv_store_s"] = (
                float(profile.get("kv_store_setup_s", 0.0)) +
                float(profile.get("kv_store_forward_s", 0.0))
            )
        self.last_chunk_profile = profile

    def _log_chunk_profile(self, profile: dict[str, Any], input_frames: int, output_frames: int) -> None:
        elapsed = float(profile["total_server_chunk_s"])
        if self.settings.profile_timings:
            print(
                f"#####[STREAM] chunk={profile['chunk_idx']} in_frames={input_frames} "
                f"out_frames={output_frames} elapsed={elapsed:.3f}s "
                f"vae_enc={float(profile.get('vae_encode_s', 0.0)):.3f}s "
                f"dit={float(profile.get('dit_denoise_s', 0.0)):.3f}s "
                f"kv_store={float(profile.get('kv_store_s', 0.0)):.3f}s "
                f"vae_dec={float(profile.get('vae_decode_s', 0.0)):.3f}s "
                f"jpeg={float(profile.get('jpeg_encode_s', 0.0)):.3f}s"
            )

            print(
                f"#####[STREAM-QWAIT] chunk={profile['chunk_idx']} "
                f"enc={float(profile.get('q_wait_encode_s', 0.0)) * 1000:.0f}ms "
                f"dit={float(profile.get('q_wait_dit_s', 0.0)) * 1000:.0f}ms "
                f"dec={float(profile.get('q_wait_decode_s', 0.0)) * 1000:.0f}ms "
                f"pseudo={float(profile.get('q_wait_pseudo_s', 0.0)) * 1000:.0f}ms "
                f"post={float(profile.get('q_wait_postprocess_s', 0.0)) * 1000:.0f}ms",
                flush=True,
            )
        else:
            print(
                f"#####[STREAM] chunk={profile['chunk_idx']} in_frames={input_frames} "
                f"out_frames={output_frames} elapsed={elapsed:.3f}s profile_timings=off"
            )

    @staticmethod
    def _align_source_metas(
        source_metas: list[dict[str, Any]],
        output_count: int,
    ) -> list[dict[str, Any]]:
        if not source_metas:
            return [{} for _ in range(output_count)]
        if len(source_metas) >= output_count:
            return source_metas[-output_count:]
        return [source_metas[-1]] * output_count

    def _start_async_workers(self) -> None:
        if self._encode_queue is not None:
            return
        self._encode_queue = queue.Queue(maxsize=4)
        self._dit_queue = queue.Queue(maxsize=4)
        self._decode_queue = queue.Queue(maxsize=4)
        self._pseudo_queue = queue.Queue(maxsize=4)
        self._postprocess_queue = queue.Queue(maxsize=4)
        self._result_queue = queue.Queue()
        use_streams = torch.cuda.is_available()

        def _stream_for(module_or_device, **kw):
            # Streams must live on the device their kernels run on; a cuda:0
            # stream running cuda:1 work raises cudaErrorInvalidResourceHandle.
            if not use_streams:
                return None
            dev = module_or_device
            if isinstance(dev, torch.nn.Module):
                dev = _module_device(dev)
            dev = torch.device(dev)
            if dev.type != "cuda":
                return None
            return torch.cuda.Stream(device=dev, **kw)

        # One VAE stream per device permits temporary allocator blocks to be
        # reused across encode/decode. Locks prevent their Python submissions
        # from interleaving; dependency waits stay outside these locks.
        self._shared_vae_stream = os.environ.get("JOYAI_SHARED_VAE_STREAM", "0") == "1"
        self._vae_submission_locks = self.runtime.vae_submission_locks if self._shared_vae_stream else {}
        _vae_streams = self.runtime.vae_worker_streams if self._shared_vae_stream else {}

        def _vae_stream_for(module):
            key = str(_canonical_device(_module_device(module)))
            if key not in _vae_streams:
                _vae_streams[key] = _stream_for(module)
                self._vae_submission_locks[key] = threading.RLock()
            return _vae_streams[key]

        self._encode_stream = _vae_stream_for(self.pipeline.vae) if self._shared_vae_stream else _stream_for(self.pipeline.vae)
        _dec_dev = _module_device(self.decode_vae)
        _bg = _stream_for(_dec_dev)
        self._decode_stream = _bg
        _ps_dev = _module_device(self.pseudo_encode_vae)
        self._pseudo_stream = _bg if _canonical_device(_ps_dev) == _canonical_device(_dec_dev) else _stream_for(_ps_dev)
        if self._shared_vae_stream:
            self._decode_stream = _vae_stream_for(self.decode_vae)
            self._pseudo_stream = _vae_stream_for(self.pseudo_encode_vae)
        self._post_stream = _stream_for(self.postprocess_device)
        self._dit_stream = _stream_for(self.device, priority=-1)
        self._workers = [
            threading.Thread(target=self._encode_worker, name="joyomni-vae-encode", daemon=True),
            threading.Thread(target=self._dit_worker, name="joyomni-dit-denoise", daemon=True),
            threading.Thread(target=self._decode_worker, name="joyomni-vae-decode", daemon=True),
            threading.Thread(target=self._pseudo_worker, name="joyomni-pseudo-encode", daemon=True),
            threading.Thread(target=self._postprocess_worker, name="joyomni-postprocess", daemon=True),
        ]
        for worker in self._workers:
            worker.start()

    def _vae_submission(self, module):
        # Synchronous startup/warm-up has no worker streams and needs no lock.
        if not getattr(self, "_shared_vae_stream", False):
            return nullcontext()
        return self._vae_submission_locks[str(_canonical_device(_module_device(module)))]

    def _stop_async_workers(self) -> None:
        if self._encode_queue is None:
            return
        try:
            self._encode_queue.put(None, timeout=0.1)
        except queue.Full:
            if self._worker_error is None:
                self._encode_queue.put(None)
        for worker in self._workers:
            worker.join()
        self._encode_queue = None
        self._dit_queue = None
        self._decode_queue = None
        self._pseudo_queue = None
        self._postprocess_queue = None
        self._result_queue = None
        self._workers = []

    def _set_worker_error(self, exc: BaseException) -> None:
        import traceback as _tb
        print(f"#####[STREAM-WORKER-ERROR] {exc!r}", flush=True)
        _tb.print_exc()
        with self._worker_error_lock:
            if self._worker_error is None:
                self._worker_error = exc
        with self._pseudo_latent_condition:
            self._pseudo_latent_condition.notify_all()

    def _raise_worker_error_if_needed(self) -> None:
        with self._worker_error_lock:
            exc = self._worker_error
            self._worker_error = None
        if exc is not None:
            raise RuntimeError(f"async streaming worker failed: {exc!r}") from exc

    @staticmethod
    def _queue_size(q: queue.Queue | None) -> int | None:
        if q is None:
            return None
        try:
            return q.qsize()
        except NotImplementedError:
            return None

    def _set_debug_state(self, worker: str, state: str, chunk_idx: int | None = None) -> None:
        with self._debug_lock:
            self._debug_worker_states[worker] = {
                "state": state,
                "chunk_idx": chunk_idx,
                "at": time.time(),
            }

    def _inc_debug_counter(self, key: str, amount: int = 1) -> None:
        with self._debug_lock:
            self._debug_counters[key] = int(self._debug_counters.get(key, 0)) + amount

    def debug_snapshot(self) -> dict[str, Any]:
        now = time.time()
        with self._debug_lock:
            worker_states = {
                name: {
                    **state,
                    "age_s": now - float(state.get("at", now)),
                }
                for name, state in self._debug_worker_states.items()
            }
            counters = dict(self._debug_counters)
        return {
            "age_s": now - self._debug_started_at,
            "initialized": self.initialized,
            "has_ref_image": self.ref_image is not None,
            "chunk_idx_next": self.chunk_idx,
            "pending_frames": len(self.pending_frames),
            "frames_per_next_chunk": self.frames_per_next_chunk,
            "raw_prompt": self.raw_prompt,
            "prompt": self.prompt,
            "prompt_control": {"requested": self._prompt_request, "applied": self._prompt_revision,
                               "status": self._prompt_status, "error": self._prompt_error,
                               "encode_s": self._prompt_encode_s},
            "cache": self._cache_stats,
            "queues": {
                "encode": self._queue_size(self._encode_queue),
                "dit": self._queue_size(self._dit_queue),
                "decode": self._queue_size(self._decode_queue),
                "pseudo": self._queue_size(self._pseudo_queue),
                "postprocess": self._queue_size(self._postprocess_queue),
                "result": self._queue_size(self._result_queue),
            },
            "queue_maxsize": {
                "encode": self._encode_queue.maxsize if self._encode_queue is not None else None,
                "dit": self._dit_queue.maxsize if self._dit_queue is not None else None,
                "decode": self._decode_queue.maxsize if self._decode_queue is not None else None,
                "pseudo": self._pseudo_queue.maxsize if self._pseudo_queue is not None else None,
                "postprocess": self._postprocess_queue.maxsize if self._postprocess_queue is not None else None,
                "result": self._result_queue.maxsize if self._result_queue is not None else None,
            },
            "workers": {
                worker.name: {
                    "alive": worker.is_alive(),
                    "state": worker_states.get(worker.name.replace("joyomni-", ""), {}),
                }
                for worker in self._workers
            },
            "worker_states": worker_states,
            "counters": counters,
            "pseudo_latent_chunk_idx": self._pseudo_latent_chunk_idx,
            "last_chunk_profile": self.last_chunk_profile,
            "shared_vae_stream": getattr(self, "_shared_vae_stream", False),
            "devices": {
                "dit": str(self.device),
                "vae_encode": str(_module_device(self.pipeline.vae)),
                "vae_decode": str(_module_device(self.decode_vae)),
                "vae_pseudo": str(_module_device(self.pseudo_encode_vae)),
                "postprocess": str(self.postprocess_device),
            },
        }

    def _submit_async_chunk(
        self,
        source_frames: list[Image.Image],
        source_metas: list[dict[str, Any]],
        valid_count: int | None = None,
    ) -> None:
        if self._encode_queue is None:
            self._start_async_workers()
        assert self._encode_queue is not None
        chunk_idx = self.chunk_idx

        frozen_anchor_id = self._update_static_anchor(chunk_idx, source_frames)
        job = _ChunkJob(
            chunk_idx=chunk_idx,
            source_frames=source_frames,
            source_metas=source_metas,
            profile=self._new_profile(chunk_idx, len(source_frames)),
            total_started=time.perf_counter(),
            frozen_anchor_id=frozen_anchor_id,
            valid_count=valid_count,
        )
        self._set_debug_state("submit", "put_encode", chunk_idx)

        while True:
            self._raise_worker_error_if_needed()
            try:
                self._encode_queue.put(job, timeout=1.0)
                break
            except queue.Full:
                continue
        self._inc_debug_counter("submitted_chunks")
        self.chunk_idx += 1

    def work_status(self):
        """Small, lock-consistent snapshot for the session watchdog."""
        with self._debug_lock:
            counters = self._debug_counters
            pending = int(counters.get("submitted_chunks", 0)) - int(counters.get("returned_results", 0))
            progress = tuple(int(counters.get(k, 0)) for k in (
                "encoded_chunks", "denoised_chunks", "decoded_pixel_chunks",
                "pseudo_encoded_chunks", "postprocessed_chunks", "returned_results"))
        return pending, progress

    def inflight_chunks(self) -> int:
        with self._debug_lock:
            return int(self._debug_counters.get("submitted_chunks", 0)) - int(
                self._debug_counters.get("returned_results", 0)
            )

    def _drain_async_results(self) -> list[StreamingChunkResult]:
        if self._result_queue is None:
            return []
        results: list[StreamingChunkResult] = []
        while True:
            try:
                results.append(self._result_queue.get_nowait())
            except queue.Empty:
                break
        if results:
            self._inc_debug_counter("returned_results", len(results))
        return results

    def wait_async_result(self, timeout: float = 0.05) -> StreamingChunkResult | None:
        self._raise_worker_error_if_needed()
        if self._result_queue is None:
            time.sleep(timeout)
            return None
        try:
            result = self._result_queue.get(timeout=timeout)
            self._inc_debug_counter("returned_results")
            return result
        except queue.Empty:
            self._raise_worker_error_if_needed()
            return None

    def _encode_worker(self) -> None:
        assert self._encode_queue is not None
        assert self._dit_queue is not None
        while True:
            self._set_debug_state("vae-encode", "wait_encode_queue")
            _q_started = time.perf_counter()
            job = self._encode_queue.get()
            if job is None:
                self._set_debug_state("vae-encode", "stop")
                self._dit_queue.put(None)
                return
            self._record_queue_wait(job.profile, "q_wait_encode_s", _q_started)
            try:
                self._set_debug_state("vae-encode", "encode", job.chunk_idx)
                started = self._timer_start()
                _enc_stream_ctx = (
                    torch.cuda.stream(self._encode_stream)
                    if self._encode_stream is not None else nullcontext()
                )
                with _enc_stream_ctx:
                    ref_chunk_latent = self._encode_reference_chunk(
                        job.source_frames,
                        profile=job.profile,
                        chunk_idx=job.chunk_idx,
                        source_job=job,
                    )
                    ready = _record_ready_event()
                self._timer_record(job.profile, "reference_prepare_s", started)
                self._set_debug_state("vae-encode", "put_dit_queue", job.chunk_idx)
                self._dit_queue.put(_EncodedChunk(job=job, ref_chunk_latent=ref_chunk_latent, ready_event=ready))
                self._inc_debug_counter("encoded_chunks")
            except BaseException as exc:
                self._set_debug_state("vae-encode", "error", job.chunk_idx)
                self._set_worker_error(exc)
                self._dit_queue.put(None)
                return

    def _dit_worker(self) -> None:
        assert self._dit_queue is not None
        assert self._decode_queue is not None
        while True:
            self._set_debug_state("dit-denoise", "wait_dit_queue")
            _q_started = time.perf_counter()
            encoded = self._dit_queue.get()
            if encoded is None:
                self._set_debug_state("dit-denoise", "stop")
                self._decode_queue.put(None)
                return
            self._record_queue_wait(encoded.job.profile, "q_wait_dit_s", _q_started)
            try:
                with self.runtime.dit_lock:
                    self._set_debug_state("dit-denoise", "denoise", encoded.job.chunk_idx)
                    _dit_stream_ctx = (
                        torch.cuda.stream(self._dit_stream)
                        if self._dit_stream is not None else nullcontext()
                    )
                    with _dit_stream_ctx:
                        _consume_on_current_stream(encoded.ready_event, encoded.ref_chunk_latent)
                        self._apply_prepared_prompt(encoded.job.chunk_idx, encoded.job.profile)
                        current_chunk_latents = self._denoise_chunk(
                            encoded.ref_chunk_latent,
                            profile=encoded.job.profile,
                            chunk_idx=encoded.job.chunk_idx,
                            frozen_anchor_id=encoded.job.frozen_anchor_id,
                        )
                        self._evict_after_store(
                            encoded.job.chunk_idx,
                            frozen_anchor_id=encoded.job.frozen_anchor_id,
                        )
                        self._refresh_sink_and_measure(encoded.job.chunk_idx, encoded.job.profile)
                        _dit_ready = _record_ready_event()
                self._set_debug_state("dit-denoise", "put_decode_queue", encoded.job.chunk_idx)
                self._decode_queue.put(
                    _DenoisedChunk(job=encoded.job, current_chunk_latents=current_chunk_latents, ready_event=_dit_ready)
                )
                self._inc_debug_counter("denoised_chunks")
            except BaseException as exc:
                self._set_debug_state("dit-denoise", "error", encoded.job.chunk_idx)
                self._set_worker_error(exc)
                self._decode_queue.put(None)
                return

    def _decode_worker(self) -> None:
        assert self._decode_queue is not None
        assert self._pseudo_queue is not None
        assert self._postprocess_queue is not None
        while True:
            self._set_debug_state("vae-decode", "wait_decode_queue")
            _q_started = time.perf_counter()
            denoised = self._decode_queue.get()
            if denoised is None:
                self._set_debug_state("vae-decode", "stop")
                self._pseudo_queue.put(None)
                self._postprocess_queue.put(None)
                return
            self._record_queue_wait(denoised.job.profile, "q_wait_decode_s", _q_started)
            try:
                self._set_debug_state("vae-decode", "decode_pixels", denoised.job.chunk_idx)
                _dec_stream_ctx = (
                    torch.cuda.stream(self._decode_stream)
                    if self._decode_stream is not None else nullcontext()
                )
                with _dec_stream_ctx:
                    _consume_on_current_stream(denoised.ready_event, denoised.current_chunk_latents)
                    decoded_pixels = self._decode_chunk_pixels(
                        denoised.current_chunk_latents,
                        profile=denoised.job.profile,
                        chunk_idx=denoised.job.chunk_idx,
                    )
                    _dec_ready = _record_ready_event()

                self._set_debug_state("vae-decode", "put_pseudo_queue", denoised.job.chunk_idx)
                self._pseudo_queue.put(
                    _DecodedPixelsChunk(job=denoised.job, decoded_pixels=decoded_pixels, ready_event=_dec_ready)
                )
                self._set_debug_state("vae-decode", "put_postprocess_queue", denoised.job.chunk_idx)
                self._postprocess_queue.put(
                    _DecodedPixelsChunk(job=denoised.job, decoded_pixels=decoded_pixels, ready_event=_dec_ready)
                )
                self._inc_debug_counter("decoded_pixel_chunks")
            except BaseException as exc:
                self._set_debug_state("vae-decode", "error", denoised.job.chunk_idx)
                self._set_worker_error(exc)
                self._pseudo_queue.put(None)
                self._postprocess_queue.put(None)
                return

    def _pseudo_worker(self) -> None:
        assert self._pseudo_queue is not None
        while True:
            self._set_debug_state("pseudo-encode", "wait_pseudo_queue")
            _q_started = time.perf_counter()
            decoded = self._pseudo_queue.get()
            if decoded is None:
                self._set_debug_state("pseudo-encode", "stop")
                return
            self._record_queue_wait(decoded.job.profile, "q_wait_pseudo_s", _q_started)
            try:
                self._set_debug_state("pseudo-encode", "pseudo_encode", decoded.job.chunk_idx)
                _ps_stream_ctx = (
                    torch.cuda.stream(self._pseudo_stream)
                    if self._pseudo_stream is not None else nullcontext()
                )
                with _ps_stream_ctx:
                    _consume_on_current_stream(decoded.ready_event, decoded.decoded_pixels,
                                               decoded.job.source_pixels)
                    self._encode_next_decode_pseudo_latent(
                        decoded.decoded_pixels,
                        profile=decoded.job.profile,
                        chunk_idx=decoded.job.chunk_idx,
                    )
                self._inc_debug_counter("pseudo_encoded_chunks")
            except BaseException as exc:
                self._set_debug_state("pseudo-encode", "error", decoded.job.chunk_idx)

                self._set_worker_error(exc)
                return

    def _postprocess_worker(self) -> None:
        assert self._postprocess_queue is not None
        assert self._result_queue is not None
        while True:
            self._set_debug_state("postprocess", "wait_postprocess_queue")
            _q_started = time.perf_counter()
            decoded = self._postprocess_queue.get()
            if decoded is None:
                self._set_debug_state("postprocess", "stop")
                return
            self._record_queue_wait(decoded.job.profile, "q_wait_postprocess_s", _q_started)
            try:
                self._set_debug_state("postprocess", "pack_frames", decoded.job.chunk_idx)
                _pp_stream_ctx = (
                    torch.cuda.stream(self._post_stream)
                    if self._post_stream is not None else nullcontext()
                )
                with _pp_stream_ctx:
                    _consume_on_current_stream(decoded.ready_event, decoded.decoded_pixels)
                    packed = self._pack_decoded_pixels(
                        decoded.decoded_pixels,
                        profile=decoded.job.profile,
                        chunk_idx=decoded.job.chunk_idx,
                        source_pixels=decoded.job.source_pixels,
                    )

                n_out = len(packed)
                self._finish_chunk_profile(
                    decoded.job.profile,
                    decoded.job.total_started,
                )
                self._log_chunk_profile(
                    decoded.job.profile,
                    len(decoded.job.source_frames),
                    n_out,
                )
                self._set_debug_state("postprocess", "put_result_queue", decoded.job.chunk_idx)
                self._result_queue.put(
                    StreamingChunkResult(
                        jpegs=packed,
                        profile=decoded.job.profile,
                        source_metas=self._align_source_metas(
                            decoded.job.source_metas,
                            n_out,
                        ),
                        elapsed=float(decoded.job.profile.get("total_server_chunk_s", 0.0)),
                        valid_count=decoded.job.valid_count,
                    )
                )
                self._inc_debug_counter("postprocessed_chunks")
            except BaseException as exc:
                self._set_debug_state("postprocess", "error", decoded.job.chunk_idx)
                self._set_worker_error(exc)
                return

    @torch.no_grad()
    def _encode_reference_chunk(
        self,
        source_frames: list[Image.Image],
        *,
        profile: dict[str, Any] | None = None,
        chunk_idx: int | None = None,
        source_job=None,
    ) -> torch.Tensor:
        chunk_idx = self.chunk_idx if chunk_idx is None else chunk_idx
        if chunk_idx == 0:
            started = self._timer_start()
            source_window = self._frames_to_tensor(source_frames[:1], source_job=source_job)
            if profile is not None:
                self._timer_record(profile, "frames_to_tensor_s", started)
        else:
            if self.prev_source_frame is None:
                raise RuntimeError("Missing previous source frame for streaming VAE encode.")
            if len(source_frames) != self.ffactor_t:
                raise ValueError(
                    f"Expected {self.ffactor_t} frames after the first chunk, got {len(source_frames)}."
                )
            started = self._timer_start()
            new_frames = self._frames_to_tensor(source_frames, source_job=source_job)
            prev = self.prev_source_frame
            if prev.device != new_frames.device:
                prev = prev.to(new_frames.device)
            source_window = torch.cat([prev, new_frames], dim=2)
            if profile is not None:
                self._timer_record(profile, "frames_to_tensor_s", started)
        self.prev_source_frame = source_window[:, :, -1:].detach().clone()
        encode_device = _module_device(self.pipeline.vae)
        source_window = source_window.to(device=encode_device, dtype=self.target_dtype)

        _vc = _vae_compile_module()
        _vc.maybe_setup_encode(self.pipeline.vae)
        source_window = _vc.prep_input(source_window)
        started = self._timer_start(encode_device)

        _enc_dev_type = torch.device(encode_device).type
        _enc_ctx = _autocast_ctx(_enc_dev_type, self.vae_dtype, self.vae_autocast_enabled)
        with self._vae_submission(self.pipeline.vae), _enc_ctx, self.runtime.stage_timings.measure('vae_encode', encode_device, chunk_idx):
            ref_latent = self.pipeline._sample_vae_latents(
                source_window,
                enable_denormalization=self.enable_denormalization,
            )
        if profile is not None:
            self._timer_record(profile, "vae_encode_s", started)
        ref_latent = ref_latent[:, :, -self.chunk_size:].to(device=self.device, dtype=self.target_dtype)
        return ref_latent

    @torch.no_grad()
    def _wait_decode_pseudo_latent(
        self,
        chunk_idx: int,
        decode_device: torch.device,
    ) -> torch.Tensor:
        while True:
            with self._pseudo_latent_condition:
                if self._pseudo_latent_chunk_idx == chunk_idx and self._pseudo_latent is not None:
                    pseudo_latent = self._pseudo_latent
                    ready = getattr(self, "_pseudo_latent_ready_event", None)
                    self._pseudo_latent = None
                    self._pseudo_latent_chunk_idx = None
                    self._pseudo_latent_ready_event = None
                    break
                self._pseudo_latent_condition.wait(timeout=0.05)
            self._raise_worker_error_if_needed()
        _consume_on_current_stream(ready, pseudo_latent)
        return pseudo_latent.to(device=decode_device, dtype=self.target_dtype)

    def _store_decode_pseudo_latent(self, chunk_idx: int, pseudo_latent: torch.Tensor) -> None:
        ready = _record_ready_event()
        with self._pseudo_latent_condition:
            self._pseudo_latent = pseudo_latent.detach()
            self._pseudo_latent_chunk_idx = chunk_idx
            self._pseudo_latent_ready_event = ready
            self._pseudo_latent_condition.notify_all()

    @torch.no_grad()
    def _decode_chunk_pixels(
        self,
        current_chunk_latents: torch.Tensor,
        *,
        profile: dict[str, Any] | None = None,
        chunk_idx: int | None = None,
    ) -> torch.Tensor:
        chunk_idx = self.chunk_idx if chunk_idx is None else chunk_idx
        decode_vae = self.decode_vae
        vae_device = _module_device(decode_vae)
        chunk_lat_flat = current_chunk_latents
        if self.enable_denormalization:
            chunk_lat_flat = self.pipeline.denormalize_latents(chunk_lat_flat)

        vae_device_type = vae_device.type
        chunk_lat_flat = chunk_lat_flat.to(vae_device)

        if chunk_idx > 0:
            self._set_debug_state("vae-decode", "wait_pseudo_latent", chunk_idx)
            pseudo_latent = self._wait_decode_pseudo_latent(chunk_idx, vae_device)
            self._set_debug_state("vae-decode", "cat_pseudo_latent", chunk_idx)
            decode_input = torch.cat([pseudo_latent, chunk_lat_flat], dim=2)
            del pseudo_latent
        else:
            decode_input = chunk_lat_flat

        _vc = _vae_compile_module()
        _vc.maybe_setup_decode(decode_vae)
        decode_input = _vc.prep_input(decode_input)

        vae_ctx = _autocast_ctx(vae_device_type, self.vae_dtype, self.vae_autocast_enabled)
        with self._vae_submission(decode_vae), vae_ctx, self.runtime.stage_timings.measure('vae_decode', vae_device, chunk_idx):
            started = self._timer_start(vae_device)
            chunk_decoded = decode_vae.decode(decode_input, return_dict=False)[0]
            if profile is not None:
                self._timer_record(profile, "vae_decode_s", started)

        if chunk_idx > 0:
            window_pixels = self.chunk_size * self.ffactor_t
            chunk_decoded = chunk_decoded[:, :, -window_pixels:]
        return chunk_decoded.detach()

    @torch.no_grad()
    def _encode_next_decode_pseudo_latent(
        self,
        decoded_pixels: torch.Tensor,
        *,
        profile: dict[str, Any] | None = None,
        chunk_idx: int,
    ) -> None:
        pseudo_vae = self.pseudo_encode_vae
        pseudo_device = _module_device(pseudo_vae)
        pseudo_device_type = pseudo_device.type
        prev_pixels = decoded_pixels[:, :, -1:].detach().to(device=pseudo_device, dtype=self.vae_dtype)

        _vc = _vae_compile_module()
        _vc.maybe_setup_encode(pseudo_vae)
        prev_pixels = _vc.prep_input(prev_pixels)
        vae_ctx = _autocast_ctx(pseudo_device_type, self.vae_dtype, self.vae_autocast_enabled)
        with self._vae_submission(pseudo_vae), vae_ctx, self.runtime.stage_timings.measure('vae_pseudo', pseudo_device, chunk_idx):
            pseudo_enc = pseudo_vae.encode(prev_pixels)
            if hasattr(pseudo_enc, "latent_dist"):
                pseudo_latent = pseudo_enc.latent_dist.sample()
            else:
                pseudo_latent = pseudo_enc
        self._set_debug_state("postprocess", "store_pseudo_latent", chunk_idx)
        self._store_decode_pseudo_latent(chunk_idx + 1, pseudo_latent)
        del prev_pixels, pseudo_enc, pseudo_latent

    @torch.no_grad()
    def _pack_decoded_pixels(
        self,
        decoded_pixels: torch.Tensor,
        *,
        profile: dict[str, Any] | None = None,
        chunk_idx: int,
        source_pixels=None,
    ) -> list[bytes]:
        post_device = self.postprocess_device

        chunk_decoded = decoded_pixels.to(device=post_device, dtype=torch.float32)
        frames_u8 = (
            (chunk_decoded / 2 + 0.5)
            .clamp(0, 1)
            .mul_(255.0)
            .round_()
            .clamp_(0, 255)
            .to(torch.uint8)[0]
            .permute(1, 2, 3, 0)
            .contiguous()
        )
        from .shader_strength import mix_shader_presentation
        strength = (profile or {}).get('shader_strength', {})
        frames_u8 = mix_shader_presentation(frames_u8, source_pixels, strength)
        if profile is not None:
            profile['shader_presentation'] = {
                'mix': strength.get('outputMix', 0),
                'previousMix': strength.get('outputMixPrevious', 0),
                'rawSource': strength.get('outputMix') == strength.get('outputMixPrevious') == 1,
                'gpuMix': frames_u8.is_cuda,
                'additionalDownloadBytes': 0,
                'source': 'matched native input before RIFE/upscaling',
            }
        if (self.runtime.lossless_output or self.settings.output_codec == "h264"
                or getattr(self.runtime, "raw_presentation_output", False)):
            return list(frames_u8.cpu().numpy())

        quality = max(1, min(100, int(self.runtime.output_quality)))
        global _NVJPEG_OK
        if _NVJPEG_OK and frames_u8.device.type == "cuda":
            started = time.perf_counter()
            try:
                chw = frames_u8.permute(0, 3, 1, 2).contiguous()
                torch.cuda.current_stream(chw.device).synchronize()
                encoded = _tv_encode_jpeg(list(chw.unbind(0)), quality=quality)
                torch.cuda.synchronize(chw.device)
                jpegs = [bytes(e.cpu().numpy()) for e in encoded]
                if profile is not None:
                    profile["jpeg_encode_s"] = float(profile.get("jpeg_encode_s", 0.0)) + (
                        time.perf_counter() - started
                    )
                return jpegs
            except Exception as exc:  # noqa: BLE001
                _NVJPEG_OK = False
                print(f"#####[STREAM] nvJPEG encode failed, falling back to cv2: {exc!r}", flush=True)

        arr = frames_u8.cpu().numpy()
        started = time.perf_counter()

        import cv2
        enc_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        jpegs = []
        for t in range(arr.shape[0]):
            bgr = cv2.cvtColor(arr[t], cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, enc_params)
            if not ok:
                raise RuntimeError(f"cv2.imencode failed for frame {t} of chunk {chunk_idx}")
            jpegs.append(buf.tobytes())
        if profile is not None:
            profile["jpeg_encode_s"] = float(profile.get("jpeg_encode_s", 0.0)) + (time.perf_counter() - started)
        return jpegs

    def _next_selected_chunk_ids(self, chunk_idx: int | None = None) -> list[int]:
        chunk_idx = self.chunk_idx if chunk_idx is None else chunk_idx
        next_total = chunk_idx + 2
        window = self.pipeline._get_last_chunk_window(
            total_latent_frames=next_total,
            chunk_size=self.chunk_size,
            window_size=self.local_window_size,
            global_sink_chunk=self.global_sink_chunk,
        )
        return list(window["selected_chunk_ids"])

    def _resize_frame(self, frame: Image.Image) -> Image.Image:
        frame = frame.convert("RGB")
        if frame.size == (self.settings.width, self.settings.height):
            return frame
        resampling = getattr(Image, "Resampling", Image).BICUBIC
        return frame.resize((self.settings.width, self.settings.height), resampling)

    def _frames_to_tensor(self, frames: list[Image.Image], *, source_job=None) -> torch.Tensor:
        arrays = [np.asarray(self._resize_frame(frame), dtype=np.uint8) for frame in frames]
        u8 = torch.from_numpy(np.stack(arrays, axis=0))
        if torch.cuda.is_available():
            encode_device = _module_device(self.pipeline.vae)
            if encode_device.type == "cuda":
                u8 = u8.to(encode_device, non_blocking=True)
        if source_job is not None:
            source_job.source_pixels = u8
        pixel = rearrange(u8, "t h w c -> 1 c t h w").to(torch.float32)
        return pixel / 127.5 - 1.0
