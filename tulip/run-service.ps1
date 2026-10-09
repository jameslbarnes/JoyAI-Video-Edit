param([int]$Gpu=0, [int]$Port=7873, [int]$TextGpu=-1, [int]$SinkRefreshChunks=0, [string]$RifeFluxRoot='', [int]$UpscaleGpu=-1, [switch]$ResidentText, [switch]$SingleGpu96GB=($env:JOYAI_SINGLE_GPU_96GB -eq '1'), [switch]$PrintConfiguration)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
if($SingleGpu96GB){
  if($Gpu -ne 0 -or $TextGpu -gt 0 -or $UpscaleGpu -gt 0){throw 'Single-GPU mode requires physical GPU 0 for every component'}
  $gpuRows=@(& nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits)
  if($LASTEXITCODE -ne 0 -or $gpuRows.Count -ne 1){throw 'Single-GPU mode requires exactly one NVIDIA GPU'}
  $memory=$gpuRows[0].Split(',')
  if([double]$memory[0] -lt 90000){throw 'Single-GPU mode requires a 96GB card'}
  if(-not $PrintConfiguration -and [double]$memory[1] -lt 70000){throw 'Single-GPU mode requires at least 70,000 MiB free; release the previous primary through SDK first'}
  if(-not $PrintConfiguration -and (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory -lt 20*1024*1024){throw 'Cold loading requires at least 20 GiB free host RAM'}
  if ($env:JOYAI_POWER_LIMIT_W -and -not $PrintConfiguration) {
    $powerLimit = [int]$env:JOYAI_POWER_LIMIT_W
    if ($powerLimit -lt 150 -or $powerLimit -gt 600) { throw 'JOYAI_POWER_LIMIT_W must be between 150 and 600' }
    $observedPower = & nvidia-smi -i 0 --query-gpu=power.limit --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0) { throw 'Could not read the single-GPU power limit' }
    if ([double]$observedPower -ne $powerLimit) {
      & nvidia-smi -i 0 -pl $powerLimit | Out-Null
      if ($LASTEXITCODE -ne 0) { throw 'Could not apply the configured single-GPU power limit; set it from an elevated operator session before starting the limited-user runtime' }
      $observedPower = & nvidia-smi -i 0 --query-gpu=power.limit --format=csv,noheader,nounits
      if ($LASTEXITCODE -ne 0 -or [double]$observedPower -ne $powerLimit) { throw 'GPU power-limit readback did not match the configured limit' }
    }
  }
  $TextGpu=0
  $UpscaleGpu=0
}
. (Join-Path $PSScriptRoot 'runtime-settings.ps1')
$performance = Get-PerformanceSettings (Join-Path $PSScriptRoot 'performance-settings.json')
if ($performance.compileVae) {
  $vswhere='C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
  $vsRoot=& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  if (!$vsRoot) { throw 'VAE compilation requires Visual Studio C++ Build Tools' }
  $env:PATH="$(Split-Path $vswhere);$env:PATH"
  $vcvars=Join-Path $vsRoot 'VC\Auxiliary\Build\vcvars64.bat'
  $buildEnv=& cmd.exe /d /s /c "`"$vcvars`" >nul && set"
  if ($LASTEXITCODE -ne 0) { throw 'Could not initialize the VAE compiler environment' }
  foreach($line in $buildEnv) {
    if($line -match '^([^=]+)=(.*)$') {[Environment]::SetEnvironmentVariable($matches[1],$matches[2],'Process')}
  }
  if(-not (Get-Command cl.exe -ErrorAction SilentlyContinue)){throw 'VAE compilation requires cl.exe on PATH after vcvars64'}
}
if (-not $PSBoundParameters.ContainsKey('ResidentText')) {
  $ResidentText = Get-ResidentTextSetting (Join-Path $PSScriptRoot 'runtime-settings.json')
}
Write-Host ('JoyAI prompt encoder: '+ $(if($ResidentText){'GPU resident'}else{'CPU offload'}))

if($ResidentText -and -not $SingleGpu96GB -and ($TextGpu -lt 0 -or $TextGpu -eq $Gpu)){
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
  if($UpscaleGpu -eq $Gpu -and -not $SingleGpu96GB){throw 'Use a separate GPU for output SR to preserve JoyAI headroom'}
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
if($SingleGpu96GB){$env:CUDA_HOME='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'}
$env:PATH="$env:CUDA_HOME\bin;$env:PATH"
$env:LIB="$env:CUDA_HOME\lib;$env:LIB"
$env:INCLUDE="$env:CUDA_HOME\include\targets\x64;$env:INCLUDE"
if($SingleGpu96GB){
  $env:LIB="$env:CUDA_HOME\lib\x64;$env:LIB"
  $env:INCLUDE="$env:CUDA_HOME\include;$env:INCLUDE"
}
$env:PYTHONUNBUFFERED='1'
$env:JOYOMNI_LOW_VRAM='1'
if($SingleGpu96GB){$env:JOYOMNI_LOW_VRAM='0'}
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
if($PrintConfiguration){
  $resolved=@{}
  Get-ChildItem Env: | Where-Object {$_.Name -match '^(CUDA_VISIBLE_DEVICES|JOYAI_(VAE.*DEVICE|RIFE_DEVICE|UPSCALE_DEVICE)|JOYOMNI_(TEXT_DEVICE|TE_OFFLOAD|LOW_VRAM)|TORCH_COMPILE_DISABLE|TORCHINDUCTOR_CACHE_DIR|TRITON_CACHE_DIR|CUDA_CACHE_PATH)$'} | ForEach-Object {$resolved[$_.Name]=$_.Value}
  $resolved | ConvertTo-Json
  return
}
Push-Location $repo
try {
  # A BelowNormal parent also demotes Python and its isolated frame clock.
  # Keep the complete delivery chain at Normal without using High/Realtime.
  try {
    [System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = 'Normal'
  } catch {
    Write-Warning "Could not set JoyAI delivery priority to Normal: $_"
  }
  & .venv/Scripts/python.exe -u tulip/serve.py --port $Port
  if($LASTEXITCODE -ne 0){throw 'JoyAI service failed; inspect the service log'}
} finally {Pop-Location}
