$ErrorActionPreference = 'Stop'
$source = Get-Content (Join-Path $PSScriptRoot 'run-service.ps1') -Raw
$start = $source.IndexOf('  if ($env:JOYAI_POWER_LIMIT_W')
$end = $source.IndexOf('  $TextGpu=0', $start)
$guard = [scriptblock]::Create($source.Substring($start, $end-$start))
$env:JOYAI_POWER_LIMIT_W = '550'
$PrintConfiguration = $false
function nvidia-smi {
  if ($args -contains '-pl') {
    $script:writes++
    if ($script:deny) { $global:LASTEXITCODE=1; return }
    $script:observed=550
  } else { $script:reads++; Write-Output $script:observed }
  $global:LASTEXITCODE=0
}
$script:observed=550; $script:writes=0; $script:reads=0; $script:deny=$true
& $guard
if ($script:writes -ne 0 -or $script:reads -ne 1) { throw 'Already-safe limit must not require elevation' }
$script:observed=600; $script:deny=$false
& $guard
if ($script:observed -ne 550 -or $script:reads -ne 3) { throw 'Changed limit must be verified' }
$script:observed=600; $script:deny=$true
$refused=$false
try { & $guard } catch { $refused=$true }
if (!$refused) { throw 'Unsafe limit must refuse startup when changing it fails' }
Write-Output 'Power-limit tests passed (3 cases)'
