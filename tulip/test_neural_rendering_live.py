import json, pathlib, sys, tempfile, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/'deploy'))
import numpy as np
from PIL import Image
from xvideo.serving.neural_rendering_output import NeuralRenderingOutput
root=pathlib.Path(__file__).resolve().parents[1]
exe=root/'tulip/research/video2dlssnr/out/video2dlssnr.exe'
frame=np.asarray(Image.open(root/'tulip/evidence/upscale-pixel-same-input-20260912/sr-48.png').convert('RGB'))
with tempfile.TemporaryDirectory() as td:
    path=pathlib.Path(td)/'config.json'
    cfg={'enabled':True,'executable':str(exe),'adapter':0,'intensity':1.0}
    path.write_text(json.dumps(cfg))
    nr=NeuralRenderingOutput(path)
    try:
        t=time.perf_counter(); first=nr.process([frame]); cold=time.perf_counter()-t
        assert nr.error is None, nr.status()
        assert first[0].shape==frame.shape and first[0].dtype==np.uint8
        delta=float(np.abs(first[0].astype(float)-frame.astype(float)).mean())
        assert delta>.1, delta
        pid=nr.process_handle.pid
        t=time.perf_counter(); chunk=nr.process([frame]*16); warm=time.perf_counter()-t
        assert len(chunk)==16 and nr.frames==17 and nr.process_handle.pid==pid, nr.status()
        # A crashed helper must latch off and return the original SR chunk intact.
        nr.process_handle.kill();nr.process_handle.wait()
        source=[frame]; assert nr.process(source) is source
        assert nr.error and nr.process_handle is None, nr.status()
    finally:nr.close()
    nr=NeuralRenderingOutput(path)
    try:
        nr.process([frame]);assert nr.error is None,nr.status()
        cfg['enabled']=False;path.write_text(json.dumps(cfg))
        source=[frame];assert nr.process(source) is source
        assert not nr.status()['enabled'] and nr.process_handle is None
    finally:nr.close()
    print(json.dumps({'passed':True,'singleFrameFlush':True,'persistentChunkFrames':16,'crashFallback':True,'toggleOff':True,'cold_s':cold,'warmChunk_s':warm,'warm_fps':16/warm,'meanPixelChange':delta}))
