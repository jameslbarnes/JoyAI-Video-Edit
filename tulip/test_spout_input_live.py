"""Run in the interactive show session with generation stopped."""
import json
from pathlib import Path
import sys
import time
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
from xvideo.serving.spout_input import SpoutInput
from xvideo.serving.spout_output import library


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    lib = library()
    sender = None
    reader = SpoutInput("ETHEREA input ownership test", "cuda:0")
    retained = []
    count = 0
    try:
        for w, h in [(128, 72), (160, 90)]:
            sender = lib.jsp_create(b"ETHEREA input ownership test", w, h, 24., 4, 1)
            if not sender:
                raise RuntimeError(lib.jsp_error(None).decode())
            pixels = torch.empty((h, w, 4), dtype=torch.uint8, device="cuda:0")
            event = torch.cuda.Event()
            for i in range(32):
                pixels[:h//2, :, 0] = 220
                pixels[h//2:, :, 0] = 10
                pixels[:h//2, :, 2] = 10
                pixels[h//2:, :, 2] = 220
                pixels[:, :, 1] = (i+1) % 256
                pixels[:, :, 3] = 255
                event.record()
                assert lib.jsp_submit(sender, pixels.data_ptr(), event.cuda_event, i+1) == 1
                frame = reader.capture(w, h, timeout=2)
                got = frame.tensor().cpu()
                assert (got[:h//2, :, 0] == 220).all() and (got[h//2:, :, 0] == 10).all()
                assert (got[:h//2, :, 2] == 10).all() and (got[h//2:, :, 2] == 220).all()
                assert (got[:, :, 1] == i+1).all()
                pixels.fill_(127)
                if i % 8 == 0:
                    retained.append((frame, got.clone()))
                count += 1
            for old, expected in retained:
                assert torch.equal(old.tensor().cpu(), expected), "Receiver reused a retained frame"
            with torch.cuda.device(1):
                transferred = frame.tensor("cuda:1")
                assert torch.equal(transferred.cpu(), got), "Cross-adapter transfer corrupted pixels"
            # The sender remains alive but no new pixels are published.
            try:
                reader.capture(w, h, timeout=.15)
                raise AssertionError("Stale input was accepted")
            except TimeoutError:
                pass
            lib.jsp_close(sender)
            sender = None
            # Observe disappearance so name reuse is a new source epoch.
            try:
                reader.capture(w, h, timeout=.15)
            except TimeoutError:
                pass
        print(json.dumps({"patternFrames": count, "ownership": True, "orientation": True,
                          "formatChange": True, "staleRejected": True, "crossAdapterExact": True}))
    finally:
        if sender:
            lib.jsp_close(sender)
        reader.close()
    shader = SpoutInput("Easel - FLUX-IN", "cuda:0")
    try:
        samples, first = [], None
        for i in range(60):
            frame = shader.capture(1152, 648, timeout=2)
            if first is None:
                first = frame
                frame.to_pil().save(out / "shader-spout.png")
            samples.append(shader.status())
            time.sleep(1/12)
        assert first is not None
        assert not torch.equal(first.tensor(), frame.tensor()), "Live shader did not change"
        (out / "input-live.json").write_text(json.dumps(samples, indent=2))
        print(json.dumps({"liveFrames": len(samples), "live": shader.status()}))
    finally:
        shader.close()


if __name__ == "__main__":
    main()
