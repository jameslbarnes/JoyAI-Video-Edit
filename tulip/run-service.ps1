param([int]$Gpu=0, [int]$Port=7873, [int]$TextGpu=-1, [int]$SinkRefreshChunks=0, [string]$RifeFluxRoot='', [int]$UpscaleGpu=-1, [switch]$ResidentText)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot 'runtime-settings.ps1')
$performance = Get-PerformanceSettings (Join-Path $PSScriptRoot 'performance-settings.json')
if ($performance.compileVae) {
  $vswhere='C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
  $vsRoot=& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  if (!$vsRoot) { throw 'VAE compilation requires Visual Studio C++ Build Tools' }
  $vcvars=Join-Path $vsRoot 'VC\Auxiliary\Build\vcvars64.bat'
  $buildEnv=& cmd.exe /d /s /c "`"$vcvars`" >nul && set"
  if ($LASTEXITCODE -ne 0) { throw 'Could not initialize the VAE compiler environment' }
  foreach($line in $buildEnv) {
    if($line -match '^([^=]+)=(.*)$') {[Environment]::SetEnvironmentVariable($matches[1],$matches[2],'Process')}
  }
}
if (-not $PSBoundParameters.ContainsKey('ResidentText')) {
  $ResidentText = Get-ResidentTextSetting (Join-Path $PSScriptRoot 'runtime-settings.json')
}
Write-Host ('JoyAI prompt encoder: '+ $(if($ResidentText){'GPU resident'}else{'CPU offload'}))

if($ResidentText -and ($TextGpu -lt 0 -or $TextGpu -eq $Gpu)){
  throw 'Resident BF16 text requires a separate -TextGpu; no workloads were changed'
}
if($RifeFluxRoot){$env:JOYAI_RIFE_FLUX_ROOT=(Resolve-Path -LiteralPath $RifeFluxRoot).Path}
$env:CUDA_VISIBLE_DEVICES="$Gpu"
if($TextGpu -ge 0 -and $TextGpu -ne $Gpu){
  $textFree=[double](& nvidia-smi -i $TextGpu --query-gpu=memory.free --format=csv,noheader,nounits)
  $textMinimum=4096
  if($ResidentText){$textMinimum=20480}
  if($textFree -lt $textMinimum){throw "Text-encoder GPU needs at least $textMinimum MiB free; no other workload was changed"}
  $env:CUDA_VISIBLE_DEVICES="$Gpu,$TextGpu"
  $env:JOYOMNI_TEXT_DEVICE='cuda:1'
} else {$env:JOYOMNI_TEXT_DEVICE='cuda:0'}
Remove-Item Env:JOYAI_UPSCALE_DEVICE -ErrorAction SilentlyContinue
Remove-Item Env:JOYAI_RIFE_DEVICE -ErrorAction SilentlyContinue
if($UpscaleGpu -ge 0){
  if($UpscaleGpu -eq $Gpu){throw 'Use a separate GPU for output SR to preserve JoyAI headroom'}
  $srFree=[double](& nvidia-smi -i $UpscaleGpu --query-gpu=memory.free --format=csv,noheader,nounits)
  if($srFree -lt 1024){throw 'Output SR GPU needs at least 1 GiB free; no workload was changed'}
  $visibleGpus=@($env:CUDA_VISIBLE_DEVICES.Split(','))
  if("$UpscaleGpu" -notin $visibleGpus){$visibleGpus += "$UpscaleGpu"}
  $env:CUDA_VISIBLE_DEVICES=$visibleGpus -join ','
  $env:JOYAI_UPSCALE_DEVICE='cuda:'+([array]::IndexOf($visibleGpus,"$UpscaleGpu"))
}
$env:JOYAI_RIFE_DEVICE=if($performance.rifeDevice -eq 'output' -and $env:JOYAI_UPSCALE_DEVICE){$env:JOYAI_UPSCALE_DEVICE}else{'cuda:0'}
$env:JOYAI_VAE_DEVICE='cuda:0'
if($performance.vaeDevice -eq 'output'){
  if(-not $env:JOYAI_UPSCALE_DEVICE){throw 'VAE output placement needs a separate output GPU'}
  $env:JOYAI_VAE_DEVICE=$env:JOYAI_UPSCALE_DEVICE
}
$env:JOYAI_VAE_ENCODE_DEVICE=$env:JOYAI_VAE_DEVICE
if($performance.vaeEncodeDevice -eq 'output'){
  if(-not $env:JOYAI_UPSCALE_DEVICE){throw 'VAE encode placement needs a separate output GPU'}
  $env:JOYAI_VAE_ENCODE_DEVICE=$env:JOYAI_UPSCALE_DEVICE
} elseif($performance.vaeEncodeDevice -eq 'model') {
  $env:JOYAI_VAE_ENCODE_DEVICE='cuda:0'
}
$env:JOYAI_VAE_PSEUDO_DEVICE=$env:JOYAI_VAE_DEVICE
if($performance.vaePseudoDevice -eq 'output'){
  if(-not $env:JOYAI_UPSCALE_DEVICE){throw 'VAE pseudo placement needs a separate output GPU'}
  $env:JOYAI_VAE_PSEUDO_DEVICE=$env:JOYAI_UPSCALE_DEVICE
} elseif($performance.vaePseudoDevice -eq 'model') {
  $env:JOYAI_VAE_PSEUDO_DEVICE='cuda:0'
}
$env:JOYOMNI_SINK_REFRESH_CHUNKS="$SinkRefreshChunks"
$env:CUDA_HOME=Join-Path $repo '.toolchain\Library'
$env:PATH="$env:CUDA_HOME\bin;$env:PATH"
$env:LIB="$env:CUDA_HOME\lib;$env:LIB"
$env:INCLUDE="$env:CUDA_HOME\include\targets\x64;$env:INCLUDE"
$env:PYTHONUNBUFFERED='1'
$env:JOYOMNI_LOW_VRAM='1'
# This switch affects only the separate BF16 MiMo prompt encoder, not DiT FP8.
# Set both branches explicitly: do not inherit stale shell residency settings.
$env:JOYOMNI_TE_OFFLOAD=if($ResidentText){'0'}else{'1'}
$env:JOYOMNI_FP8_IMG='1'
$env:JOYOMNI_FP8_TXT='1'
$env:JOYOMNI_FP8_FAST_ACCUM='1'
$env:JOYOMNI_SAGE_ATTN='1'
$env:JOYOMNI_SKIP_LOAD_WARMUP='1'
$env:JOYOMNI_CUDA_GRAPH='1'
$env:TORCH_COMPILE_DISABLE=if($performance.compileVae){'0'}else{'1'}
$env:JOYOMNI_FIXED_PROMPT_GRAPH=if($performance.fixedPromptGraph){'1'}else{'0'}
$env:JOYAI_COMPACT_PROMPT_GRAPHS=if($performance.compactPromptGraphs){'1'}else{'0'}
$env:JOYAI_SINGLE_PROMPT_GRAPH=if($performance.singlePromptGraph){'1'}else{'0'}
$env:JOYOMNI_TRIM_PADDED_TEXT=if($performance.trimPaddedText){'1'}else{'0'}
$env:JOYAI_SR_GRAPH=if($performance.srGraph){'1'}else{'0'}
$env:JOYAI_SHARED_VAE_STREAM=if($performance.sharedVaeStream){'1'}else{'0'}
$env:JOYOMNI_WARMUP_NATIVE_ONLY='1'
$env:JOYOMNI_COMPILE_REFERENCE='0'
$env:TORCHINDUCTOR_FX_GRAPH_CACHE='1'
$env:TORCHINDUCTOR_UNIQUE_KERNEL_NAMES='0'
$env:TORCHINDUCTOR_CACHE_DIR=Join-Path $repo 'tulip\cache\inductor'
$env:TRITON_CACHE_DIR=Join-Path $repo 'tulip\cache\triton'
$env:CUDA_CACHE_PATH=Join-Path $repo 'tulip\cache\cuda'
Push-Location $repo
try {
  & .venv/Scripts/python.exe -u tulip/serve.py --port $Port
  if($LASTEXITCODE -ne 0){throw 'JoyAI service failed; inspect the service log'}
} finally {Pop-Location}
