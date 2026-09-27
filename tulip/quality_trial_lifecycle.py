"""Run a bounded offline benchmark through the SDK-owned JoyAI lifecycle."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
sdk = root.parent / 'easel-agent-sdk'
out = Path((root / 'tulip/quality-tuning-current.txt').read_text().strip())
base_out = out
restart_only = '--restart-only' in sys.argv
optimize = '--optimize' in sys.argv
sweep = '--sweep' in sys.argv
deploy = '--deploy' in sys.argv or restart_only or optimize or sweep
if deploy:
    out = out / ('deployment-' + time.strftime('%H%M%S'))
    out.mkdir()
sys.path.insert(0, str(sdk / 'src'))
for line in (sdk / '.env').read_text(encoding='utf-8-sig').splitlines():
    line = line.strip()
    if line and not line.startswith('#') and '=' in line:
        key, value = line.removeprefix('export ').split('=', 1)
        os.environ[key.strip()] = value.strip().strip(chr(39)+chr(34))
from easel_agent.cli import _router
from easel_agent.primary_models import LivePrimaryAdapter
router = _router(dry_run=False, config_path=str(sdk / 'nanos.json'))
adapter = LivePrimaryAdapter(router.store)
backend = 'http://127.0.0.1:7860'
control = 'http://127.0.0.1:8793'

def save(name, value):
    (out / (name+'.json')).write_text(json.dumps(value, indent=2), encoding='utf-8')

def act(name, payload=None):
    result = router.dispatch(name, {'config': str(sdk/'nanos.json'),
        'source': 'user-requested-native-quality-tuning', **(payload or {})})
    save(name, result)
    print(json.dumps({'action':name,'ok':result.get('ok'),'error':result.get('error')}), flush=True)
    if not result.get('ok'): raise RuntimeError(name+' failed')
    return result

listing = adapter._request('GET', control+'/control/multistream/list')
save('before-live', listing)
tile = next(s for s in listing['streams'] if s.get('type')=='joyai')
stats = tile['runtime']['stats']
prompt = stats['prompt']
strength = stats['strength'].get('target',stats['strength']['value'])
for endpoint in ('health','debug'):
    save('before-'+endpoint, adapter._request('GET','http://127.0.0.1:7873/'+endpoint))
act('ops.hold', {'ttlMinutes':60,'reason':'Matched native resolution/interpolation/upscaling benchmark'})
removed = False
try:
    act('multistream.remove', {'provider':'joyai','force':True})
    removed = True
    result = adapter._request('POST',backend+'/api/runtimes/joyai/stop',
        {'force':True,'cleanupOrphans':False},timeout=40)
    save('stop',result)
    if result.get('process_running'): raise RuntimeError('JoyAI runtime did not stop')
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='1,0',JOYOMNI_TEXT_DEVICE='cuda:1',JOYOMNI_TE_OFFLOAD='0',
        JOYOMNI_LOW_VRAM='1',JOYOMNI_FP8_IMG='1',JOYOMNI_FP8_TXT='1',
        JOYOMNI_FP8_FAST_ACCUM='1',JOYOMNI_SAGE_ATTN='1',JOYOMNI_SKIP_LOAD_WARMUP='1',
        JOYOMNI_CUDA_GRAPH='1',JOYOMNI_SINK_REFRESH_CHUNKS='0',TORCH_COMPILE_DISABLE='1',
        TORCHINDUCTOR_UNIQUE_KERNEL_NAMES='0',PYTHONUNBUFFERED='1',
        JOYAI_RIFE_FLUX_ROOT=str(root.parent/'FluxRT'),CUDA_HOME=str(root/'.toolchain/Library'),
        TORCHINDUCTOR_CACHE_DIR=str(root/'tulip/cache/inductor'),
        TRITON_CACHE_DIR=str(root/'tulip/cache/triton'),CUDA_CACHE_PATH=str(root/'tulip/cache/cuda'))
    env.pop('JOYAI_UPSCALE_DEVICE',None)
    env['PATH'] = str(root/'.toolchain/Library/bin')+';'+env.get('PATH','')
    env['LIB'] = str(root/'.toolchain/Library/lib')+';'+env.get('LIB','')
    env['INCLUDE'] = str(root/'.toolchain/Library/include/targets/x64')+';'+env.get('INCLUDE','')
    print('Starting matched benchmark; output: '+str(out/'benchmark.log'),flush=True)
    if optimize:
        vswhere = Path('C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe')
        vsroot = subprocess.check_output([str(vswhere), '-latest', '-products', '*',
            '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property',
            'installationPath'], text=True).strip()
        vcvars = Path(vsroot)/'VC/Auxiliary/Build/vcvars64.bat'
        build_env = subprocess.check_output('cmd.exe /d /s /c ""'+str(vcvars)+'" >nul && set"',text=True)
        for line in build_env.splitlines():
            if '=' in line and not line.startswith('='):
                key,value=line.split('=',1)
                env[key]=value
        # Restore CUDA's local development paths after importing MSVC.
        env['PATH']=str(root/'.toolchain/Library/bin')+';'+env['PATH']
        env['LIB']=str(root/'.toolchain/Library/lib')+';'+env.get('LIB','')
        env['INCLUDE']=str(root/'.toolchain/Library/include/targets/x64')+';'+env.get('INCLUDE','')
        env['JOYOMNI_WARMUP_NATIVE_ONLY']='1'
        env['TORCHINDUCTOR_FX_GRAPH_CACHE']='1'
        arms=[('eager-no-profile','1','cuda:0','1'),
              ('eager-sage','1','cuda:0','0'),
              ('compiled-sage','0','cuda:0','0'),
              ('compiled-sage-split-vae','0','cuda:1','0')]
        if '--compiled-only' in sys.argv:
            arms=[arm for arm in arms if arm[0]=='compiled-sage']
        outcomes=[]
        for name,disabled,vae_device,fixed_prompt in arms:
            dest=out/name;dest.mkdir()
            env['TORCH_COMPILE_DISABLE']=disabled
            env['JOYOMNI_FIXED_PROMPT_GRAPH']=fixed_prompt
            print('Optimization arm '+name+'; log: '+str(dest/'run.log'),flush=True)
            started=time.monotonic()
            try:
                with (dest/'run.log').open('w',encoding='utf-8') as log:
                    size_args=['--sizes','840x480']
                    if '--compiled-only' in sys.argv:
                        size_args=['--sizes','840x480,1056x600,1248x720','--eager-after-first']
                    result=subprocess.run([str(root/'.venv/Scripts/python.exe'),'-u',
                        str(root/'tulip/quality_benchmark.py'),'--out',str(dest),'--frames','481',
                        *size_args,'--no-profile-timings','--vae-device',vae_device],
                        cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
                outcomes.append({'arm':name,'exit':result.returncode,'seconds':time.monotonic()-started})
            except subprocess.TimeoutExpired:
                outcomes.append({'arm':name,'timeout':True,'seconds':time.monotonic()-started})
            save('optimization-outcomes',outcomes)
            print(json.dumps(outcomes[-1]),flush=True)
        from types import SimpleNamespace
        result=SimpleNamespace(returncode=0)
    elif sweep:
        env['JOYOMNI_FIXED_PROMPT_GRAPH']='0'
        env['JOYOMNI_WARMUP_NATIVE_ONLY']='1'
        with (out/'benchmark.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([str(root/'.venv/Scripts/python.exe'),'-u',
                str(root/'tulip/quality_benchmark.py'),'--out',str(out),'--frames','257',
                '--sizes','840x480,1056x600,1248x720'],cwd=root,env=env,
                stdout=log,stderr=subprocess.STDOUT,timeout=900)
    elif restart_only:
        from types import SimpleNamespace
        result=SimpleNamespace(returncode=0)
    else:
        with (out/'benchmark.log').open('w',encoding='utf-8') as log:
            script = 'presentation_benchmark.py' if deploy else 'quality_benchmark.py'
            result = subprocess.run([str(root/'.venv/Scripts/python.exe'),'-u',str(root/'tulip'/script),
                '--out',str(base_out if deploy else out)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1500)
    print('Benchmark exit: '+str(result.returncode),flush=True)
    if deploy and not restart_only and not optimize and not sweep and result.returncode == 0:
        config_path=root/'tulip/native-quality.json'
        save('native-quality-before',json.loads(config_path.read_text()) if config_path.exists() else None)
        config={'width':864,'height':480,'fps':16,'rife_multiplier':2}
        pending=config_path.with_suffix('.pending.json');pending.write_text(json.dumps(config));pending.replace(config_path)
        save('native-quality-after',config)
finally:
    try:
        if removed:
            adapter._request('POST',backend+'/api/runtimes/joyai/start',{'wait':False},timeout=20)
            deadline=time.monotonic()+420
            while time.monotonic()<deadline:
                status=adapter._request('GET',backend+'/api/runtimes/joyai?dependencies=0',timeout=10)
                if status.get('state')=='ready': break
                if status.get('state') in ('error','stopped'): raise RuntimeError(str(status.get('last_error')))
                time.sleep(3)
            else: raise RuntimeError('JoyAI restoration startup timeout')
            save('restored-runtime',status)
            act('multistream.add',{'provider':'joyai','options':{'seedPrompt':prompt}})
            deadline=time.monotonic()+180
            while time.monotonic()<deadline:
                live=adapter._request('GET',control+'/control/multistream/list')
                tile=next((s for s in live['streams'] if s.get('type')=='joyai'),{})
                if tile.get('status')=='live' and tile.get('runtime',{}).get('stats',{}).get('generatedFrames',0)>16: break
                if tile.get('status')=='error': raise RuntimeError(str(tile))
                time.sleep(2)
            else: raise RuntimeError('JoyAI did not resume fresh output')
            act('model.primary.strength.set',{'provider':'joyai','value':strength})
            save('after-live',adapter._request('GET',control+'/control/multistream/list'))
            print('Live JoyAI restored',flush=True)
    finally:
        act('ops.resume')
