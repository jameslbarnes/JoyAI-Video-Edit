from pathlib import Path
p=Path('deploy/xvideo/models/dit/prompt_mask.py')
p.write_text('''"""Masked attention for fixed-capacity streaming prompt graphs.

The SM120 SageAttention wrapper does not honor arbitrary masks. Route masked
calls explicitly to cuDNN; unmasked calls retain their existing backend.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def masked_attention(query, key, value, mask):
    """NHD attention; True keys participate, False keys contribute nothing."""
    q, k, v = (x.transpose(1, 2) for x in (query, key, value))
    if query.is_cuda:
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    else:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return out.transpose(1, 2)
''',encoding='utf-8')
def edit(path,replacements):
 p=Path(path);b=p.read_bytes()
 for old,new in replacements:
  old,new=old.encode(),new.encode()
  if b.count(old)!=1:
   old=old.replace(b'\n',b'\r\n');new=new.replace(b'\n',b'\r\n')
  assert b.count(old)==1,(path,old[:100],b.count(old))
  b=b.replace(old,new)
 p.write_bytes(b)
edit('deploy/xvideo/models/dit/dit.py',[
 ('from xvideo.models.dit.rope import apply_rotary_emb, get_1d_rotary_pos_embed','from xvideo.models.dit.rope import apply_rotary_emb, get_1d_rotary_pos_embed\nfrom xvideo.models.dit.prompt_mask import masked_attention'),
 ('def _flash_attention4(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:\n    global _FA4_DISABLED','def _flash_attention4(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask=None) -> torch.Tensor:\n    global _FA4_DISABLED\n    if attn_mask is not None:\n        return masked_attention(query, key, value, attn_mask)'),
 ('        cached_freqs_cis: Optional[tuple] = None,\n    ) -> Tuple[torch.Tensor, torch.Tensor]:','        cached_freqs_cis: Optional[tuple] = None,\n        attention_mask: Optional[torch.Tensor] = None,\n    ) -> Tuple[torch.Tensor, torch.Tensor]:'),
 ('        attn = _flash_attention4(q, k, v)','        attn = _flash_attention4(q, k, v, attn_mask=attention_mask)'),
 ('        skip_text_stream: bool = False,\n    ) -> tuple[torch.Tensor, torch.Tensor]:','        skip_text_stream: bool = False,\n        attention_mask: Optional[torch.Tensor] = None,\n    ) -> tuple[torch.Tensor, torch.Tensor]:'),
 ('                cached_freqs_cis=cached_freqs_cis,\n            )','                cached_freqs_cis=cached_freqs_cis,\n                attention_mask=attention_mask,\n            )')])
edit('deploy/xvideo/serving/graph_runner.py',[
 ('class StreamingGraphRunner:', 'def fixed_prompt_graph_enabled() -> bool:\n    return os.environ.get("JOYOMNI_FIXED_PROMPT_GRAPH", "1").lower() in {"1", "true", "yes", "on"}\n\n\nclass StreamingGraphRunner:'),
 ('        autocast_ctx: Callable[[], Any],\n    ):','        autocast_ctx: Callable[[], Any],\n        mask_prompt: bool = False,\n    ):'),
 ('        self.txt_len = int(txt_len)','        self.txt_len = int(txt_len)\n        self.mask_prompt = mask_prompt\n        self.prompt_tokens = 0'),
 ('        self.in_current_ids = torch.full((1, latent_shape[2]), self.max_temporal_ids,','        # K layout: cached sink/previous/reference, current image and input, text.\n        self.prompt_key_prefix = self.pool_len + 2 * chunk_tokens\n        self.prompt_key_mask = (torch.ones((1, 1, 1, self.prompt_key_prefix + self.txt_len),\n                                          device=device, dtype=torch.bool) if mask_prompt else None)\n        self.in_current_ids = torch.full((1, latent_shape[2]), self.max_temporal_ids,'),
 ('                        ref_video_latent=self.in_ref_latent,','                        ref_video_latent=self.in_ref_latent,\n                        attention_mask=self.prompt_key_mask,'),
 ('        self.in_prompt.copy_(prompt_embeds.to(dtype=self.dtype))','        length = int(prompt_embeds.shape[1])\n        if (prompt_embeds.ndim != 3 or prompt_embeds.shape[0] != 1 or\n                prompt_embeds.shape[2] != self.in_prompt.shape[2] or\n                not 0 < length <= self.txt_len or (not self.mask_prompt and length != self.txt_len)):\n            raise ValueError("Prompt embeddings do not fit the captured graph capacity")\n        if self.mask_prompt:\n            self.in_prompt.zero_()\n            self.in_prompt[:, :length].copy_(prompt_embeds.to(dtype=self.dtype))\n            self.prompt_key_mask.fill_(False)\n            self.prompt_key_mask[..., :self.prompt_key_prefix + length].fill_(True)\n        else:\n            self.in_prompt.copy_(prompt_embeds.to(dtype=self.dtype))\n        self.prompt_tokens = length')])
edit('deploy/xvideo/serving/joyomni_streaming.py',[
 ('from xvideo.serving.graph_runner import GRAPH_WINDOW_CHUNKS, StreamingGraphRunner, graph_env_enabled','from xvideo.serving.graph_runner import GRAPH_WINDOW_CHUNKS, StreamingGraphRunner, graph_env_enabled, fixed_prompt_graph_enabled'),
 ('        return (\n            self.latent_h,\n            self.latent_w,\n            int(embeds.shape[1]),','        capacity = int(self.settings.max_sequence_length or self.cfg.text_token_max_length) if fixed_prompt_graph_enabled() else int(embeds.shape[1])\n        if not 0 < int(embeds.shape[1]) <= capacity:\n            raise ValueError("Prompt embeddings exceed the graph prompt capacity")\n        return (\n            self.latent_h,\n            self.latent_w,\n            capacity,'),
 ('            int(self.settings.num_inference_steps),\n        )','            int(self.settings.num_inference_steps),\n            fixed_prompt_graph_enabled(),\n        )'),
 ('                txt_len=int(self.streaming_cond_embeds.shape[1]),','                txt_len=key[2],\n                mask_prompt=fixed_prompt_graph_enabled(),'),
 ('        profile["graph_path"] = 1','        profile["graph_path"] = 1\n        profile["graph_prompt_capacity"] = runner.txt_len\n        profile["graph_prompt_tokens"] = runner.prompt_tokens\n        profile["graph_prompt_masked"] = runner.mask_prompt')])
