"""Read-only pinned source snapshots for the LongLive/VACE investigation."""

import hashlib
import json
from pathlib import Path
import requests

REV = "2aced4ded3513a76cd35f0dbfb42fbfbf5e98ab0"
ROOT = Path(__file__).parent / "research/scope-current"
LOCAL = Path(r"C:\Users\Admin\src\scope")
FILES = [
    "src/scope/core/pipelines/longlive/schema.py",
    "src/scope/core/pipelines/longlive/model.yaml",
    "src/scope/core/pipelines/longlive/pipeline.py",
    "src/scope/core/pipelines/longlive/modular_blocks.py",
    "src/scope/core/pipelines/longlive/blocks/recache_frames.py",
    "src/scope/core/pipelines/wan2_1/vace/mixin.py",
    "src/scope/core/pipelines/wan2_1/vace/models/causal_vace_model.py",
    "src/scope/core/pipelines/wan2_1/vace/models/attention_blocks.py",
    "src/scope/core/pipelines/wan2_1/vace/blocks/vace_encoding.py",
    "src/scope/core/pipelines/wan2_1/vace/utils/encoding.py",
    "src/scope/core/pipelines/wan2_1/vace/utils/weight_loader.py",
    "src/scope/core/pipelines/wan2_1/blocks/auto_prepare_latents.py",
    "src/scope/core/pipelines/wan2_1/blocks/prepare_video_latents.py",
    "src/scope/server/graph_schema.py",
    "docs/vace.md",
    "docs/api/vace.md",
]
rows = []
for name in FILES:
    url = f"https://raw.githubusercontent.com/daydreamlive/scope/{REV}/{name}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    local = LOCAL / name
    same = local.exists() and local.read_text(encoding="utf-8").replace(
        "\r\n", "\n"
    ) == response.text.replace("\r\n", "\n")
    row = dict(
        path=name,
        revision=REV,
        url=url,
        sha256=hashlib.sha256(response.content).hexdigest(),
        same_as_local=same,
    )
    rows.append(row)
    print(json.dumps(row), flush=True)
(ROOT.parent / "scope-source-manifest.json").write_text(json.dumps(rows, indent=2))
