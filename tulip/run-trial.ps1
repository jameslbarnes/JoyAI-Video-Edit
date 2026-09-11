param([int]$Gpu=0, [int]$Frames=33, [int]$Width=648, [int]$Height=360, [switch]$Optimized, [switch]$GraphsOnly,
      [string]$InputVideo='', [string]$OutputDir='', [string]$Prompt='')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
if($Optimized -and $GraphsOnly){throw 'Choose Optimized or GraphsOnly, not both'}
if($Optimized) {
  $vswhere='C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
  $vsRoot=& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  if (!$vsRoot) {throw 'Optimized mode requires existing Visual Studio C++ Build Tools'}
  $vcvars=Join-Path $vsRoot 'VC\Auxiliary\Build\vcvars64.bat'
  $buildEnv=& cmd.exe /d /s /c "`"$vcvars`" >nul && set"
  foreach($line in $buildEnv) {
    if($line -match '^([^=]+)=(.*)$') {[Environment]::SetEnvironmentVariable($matches[1],$matches[2],'Process')}
  }
}
$env:CUDA_VISIBLE_DEVICES="$Gpu"
$env:CUDA_HOME=Join-Path $repo '.toolchain\Library'
$env:PATH="$env:CUDA_HOME\bin;$env:PATH"
$env:LIB="$env:CUDA_HOME\lib;$env:LIB"
$env:INCLUDE="$env:CUDA_HOME\include\targets\x64;$env:INCLUDE"
$env:PYTHONUNBUFFERED='1'
$env:JOYOMNI_LOW_VRAM='1'
$env:JOYOMNI_FP8_IMG='1'
$env:JOYOMNI_FP8_TXT='1'
$env:JOYOMNI_FP8_FAST_ACCUM='1'
$env:JOYOMNI_SAGE_ATTN='1'
$env:JOYOMNI_SKIP_LOAD_WARMUP='1'
$env:JOYOMNI_CUDA_GRAPH=if($Optimized -or $GraphsOnly){'1'}else{'0'}
$env:TORCH_COMPILE_DISABLE=if($Optimized){'0'}else{'1'}
$env:TORCHINDUCTOR_UNIQUE_KERNEL_NAMES='0'
$env:TORCHINDUCTOR_CACHE_DIR=Join-Path $repo 'tulip\cache\inductor'
$env:TRITON_CACHE_DIR=Join-Path $repo 'tulip\cache\triton'
$env:CUDA_CACHE_PATH=Join-Path $repo 'tulip\cache\cuda'
Push-Location $repo
try {
  $trialArgs=@('-u','tulip/trial.py','--frames',"$Frames",'--width',"$Width",'--height',"$Height")
  if($InputVideo){$trialArgs+=@('--input',$InputVideo)}
  if($OutputDir){$trialArgs+=@('--output-dir',$OutputDir)}
  if($Prompt){$trialArgs+=@('--prompt',$Prompt)}
  & .venv/Scripts/python.exe @trialArgs
  if($LASTEXITCODE -ne 0){throw 'JoyAI trial failed; inspect saved metrics and log'}
} finally {Pop-Location}
