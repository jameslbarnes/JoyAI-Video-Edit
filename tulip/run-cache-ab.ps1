param([int]$Gpu=1,[string]$OutputDir,[string]$InputVideo,[int]$Frames=1025,[string]$Arms='baseline,eager,no_kv,clamp_feedback',[string]$Jobs)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$env:CUDA_VISIBLE_DEVICES="$Gpu"
$env:JOYOMNI_TEXT_DEVICE='cuda:0'
$env:CUDA_HOME=Join-Path $repo '.toolchain\Library'
$env:PATH="$env:CUDA_HOME\bin;$env:PATH"
$env:LIB="$env:CUDA_HOME\lib;$env:LIB"
$env:INCLUDE="$env:CUDA_HOME\include\targets\x64;$env:INCLUDE"
$env:JOYOMNI_LOW_VRAM='1'
$env:JOYOMNI_FP8_IMG='1'
$env:JOYOMNI_FP8_TXT='1'
$env:JOYOMNI_FP8_FAST_ACCUM='1'
$env:JOYOMNI_SAGE_ATTN='1'
$env:JOYOMNI_SKIP_LOAD_WARMUP='1'
$env:JOYOMNI_CUDA_GRAPH='1'
$env:TORCH_COMPILE_DISABLE='1'
$env:TRITON_CACHE_DIR=Join-Path $repo 'tulip/cache/triton'
Push-Location $repo
try {
  $abArgs=@('-u','tulip/cache_ab.py','--out',$OutputDir,'--input',$InputVideo,'--frames',"$Frames",'--arms') + $Arms.Split(',')
  if($Jobs){$abArgs+=@('--jobs',$Jobs)}
  & .venv/Scripts/python.exe @abArgs
  if($LASTEXITCODE -ne 0){throw 'Cache diagnostic failed; see report.json'}
} finally {Pop-Location}
