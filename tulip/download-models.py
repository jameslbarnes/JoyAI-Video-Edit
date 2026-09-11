"""Pinned, resumable downloads of only the artifacts used by the 0811 path."""

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent
ARTIFACTS = [
    (
        "jdopensource/JoyAI-Video-Edit",
        "39491dda89fd2b535777fd8644a6631ea89b857c",
        [
            "config.json",
            "meta.yaml",
            "dit/joyai_video_edit_dit_0811.pth",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ],
    ),
    (
        "XiaomiMiMo/MiMo-VL-7B-RL-2508",
        "4bfb270765825d2fa059011deb4c96fdd579be6f",
        [
            "added_tokens.json",
            "chat_template.json",
            "config.json",
            "generation_config.json",
            "merges.txt",
            "model.safetensors.index.json",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ]
        + [f"model-{i:05d}-of-00004.safetensors" for i in range(1, 5)],
    ),
]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(repo, revision, remote, directory):
    path = directory / remote.rfilename
    path.parent.mkdir(parents=True, exist_ok=True)
    size = remote.size
    expected = remote.lfs.sha256 if remote.lfs else None
    if (
        path.exists()
        and path.stat().st_size == size
        and (not expected or digest(path) == expected)
    ):
        return path
    partial = path.with_name(path.name + ".partial")
    progress = path.with_name(path.name + ".ranges.json")
    block = 16 * 1024 * 1024
    done = (
        set(json.loads(progress.read_text()))
        if progress.exists() and partial.exists()
        else set()
    )
    if not partial.exists():
        with partial.open("wb") as stream:
            stream.truncate(size)
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{remote.rfilename}"

    def fetch(start):
        end = min(start + block, size) - 1
        for attempt in range(5):
            try:
                with requests.get(
                    url,
                    params={"range_start": start},
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=(15, 60),
                ) as response:
                    response.raise_for_status()
                    payload = response.content
                    if len(payload) != end - start + 1:
                        raise RuntimeError(f"Unexpected range size for {path.name}")
                    if size > block and (
                        response.status_code != 206
                        or response.headers.get("Content-Range")
                        != f"bytes {start}-{end}/{size}"
                    ):
                        raise RuntimeError(f"Invalid range response for {path.name}")
                    with partial.open("r+b") as stream:
                        stream.seek(start)
                        stream.write(payload)
                    return start
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(attempt + 1)

    remaining = [start for start in range(0, size, block) if start not in done]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in pool.map(fetch, remaining):
            done.add(start)
            progress.write_text(json.dumps(sorted(done)))
            if len(done) % 16 == 0 or len(done) * block >= size:
                print(
                    f"{path.name}: {min(len(done) * block, size)}/{size} bytes",
                    flush=True,
                )
    actual = digest(partial)
    if expected and actual != expected:
        raise RuntimeError(f"SHA256 mismatch: {partial}")
    os.replace(partial, path)
    return path


def main():
    manifest = []
    for repo, revision, files in ARTIFACTS:
        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        directory = ROOT / "models" / repo.split("/")[1]
        for name in files:
            remote = next(s for s in info.siblings if s.rfilename == name)
            path = download(repo, revision, remote, directory)
            manifest.append(
                dict(
                    repo=repo,
                    revision=revision,
                    file=name,
                    path=str(path),
                    bytes=path.stat().st_size,
                    sha256=digest(path),
                    lfs_verified=bool(remote.lfs),
                )
            )
            (ROOT / "models-manifest.json").write_text(json.dumps(manifest, indent=2))
    print("All required artifacts verified", flush=True)


if __name__ == "__main__":
    main()
