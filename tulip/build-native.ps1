param([switch]$WithCutlass, [switch]$Sage)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$vswhere='C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$vsRoot=& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$vsRoot) {throw 'Existing Visual Studio C++ Build Tools required'}
$vcvars=Join-Path $vsRoot 'VC\Auxiliary\Build\vcvars64.bat'
$buildEnv=& cmd.exe /d /s /c "`"$vcvars`" >nul && set"
foreach($line in $buildEnv) {
  if($line -match '^([^=]+)=(.*)$') {[Environment]::SetEnvironmentVariable($matches[1],$matches[2],'Process')}
}
$env:CUDA_HOME=Join-Path $repo '.toolchain\Library'
$env:PATH="$env:CUDA_HOME\bin;$env:PATH"
$env:LIB="$env:CUDA_HOME\lib;$env:LIB"
$env:INCLUDE="$env:CUDA_HOME\include\targets\x64;$env:INCLUDE"
$env:DISTUTILS_USE_SDK='1'
$env:MSSdk='1'
$env:MAX_JOBS='4'
$env:TORCH_CUDA_ARCH_LIST='12.0'
$env:JOYOMNI_OPS_CUDA_ARCHS='120a'
$env:JOYOMNI_OPS_NO_FP8=if($WithCutlass){'0'}else{'1'}
if($WithCutlass){$env:JOYOMNI_OPS_CUTLASS_DIR=Join-Path $repo 'deploy\tmp\cutlass'}
& (Join-Path $env:CUDA_HOME 'bin\nvcc.exe') --version
$source = if($Sage){'deploy\tmp\SageAttention'}else{'deploy\joyomni_ops'}
if($Sage){$env:EXT_PARALLEL='1'}
Push-Location (Join-Path $repo $source)
try {
  & uv.exe pip install --python (Join-Path $repo '.venv\Scripts\python.exe') --no-build-isolation --no-deps .
  if($LASTEXITCODE -ne 0){throw 'JoyAI native extension build failed'}
} finally {Pop-Location}
