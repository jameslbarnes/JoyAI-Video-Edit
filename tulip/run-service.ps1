param([int]$Gpu=0, [int]$Port=7873, [int]$TextGpu=-1, [int]$SinkRefreshChunks=0, [string]$RifeFluxRoot='', [int]$UpscaleGpu=-1, [switch]$ResidentText)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
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
if($UpscaleGpu -ge 0){
  if($UpscaleGpu -eq $Gpu){throw 'Use a separate GPU for output SR to preserve JoyAI headroom'}
  $srFree=[double](& nvidia-smi -i $UpscaleGpu --query-gpu=memory.free --format=csv,noheader,nounits)
  if($srFree -lt 1024){throw 'Output SR GPU needs at least 1 GiB free; no workload was changed'}
  $visibleGpus=@($env:CUDA_VISIBLE_DEVICES.Split(','))
  if("$UpscaleGpu" -notin $visibleGpus){$visibleGpus += "$UpscaleGpu"}
  $env:CUDA_VISIBLE_DEVICES=$visibleGpus -join ','
  $env:JOYAI_UPSCALE_DEVICE='cuda:'+([array]::IndexOf($visibleGpus,"$UpscaleGpu"))
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
$env:TORCH_COMPILE_DISABLE='1'
$env:TORCHINDUCTOR_UNIQUE_KERNEL_NAMES='0'
$env:TORCHINDUCTOR_CACHE_DIR=Join-Path $repo 'tulip\cache\inductor'
$env:TRITON_CACHE_DIR=Join-Path $repo 'tulip\cache\triton'
$env:CUDA_CACHE_PATH=Join-Path $repo 'tulip\cache\cuda'
Push-Location $repo
try {
  & .venv/Scripts/python.exe -u tulip/serve.py --port $Port
  if($LASTEXITCODE -ne 0){throw 'JoyAI service failed; inspect the service log'}
} finally {Pop-Location}
