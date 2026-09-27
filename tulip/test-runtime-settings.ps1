$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'runtime-settings.ps1')
$parseErrors=$null
[System.Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'run-service.ps1'),[ref]$null,[ref]$parseErrors) | Out-Null
if($parseErrors){throw ($parseErrors | Out-String)}
$testPath=Join-Path $env:TEMP ('joyai-setting-'+[guid]::NewGuid().ToString()+'.json')
try {
  if(Get-ResidentTextSetting $testPath){throw 'Missing setting must default off'}
  '{"residentText":true}' | Set-Content -LiteralPath $testPath
  if(-not (Get-ResidentTextSetting $testPath)){throw 'True setting was not read'}
  '{"residentText":false}' | Set-Content -LiteralPath $testPath
  if(Get-ResidentTextSetting $testPath){throw 'False setting was not read'}
  '{"residentText":"false"}' | Set-Content -LiteralPath $testPath
  $rejected=$false
  try{Get-ResidentTextSetting $testPath | Out-Null}catch{$rejected=$true}
  if(-not $rejected){throw 'String setting should be rejected'}
  '{"compileVae":true,"sharedVaeStream":true,"srGraph":true,"fixedPromptGraph":false,"vaeDevice":"output","rifeDevice":"model"}' | Set-Content -LiteralPath $testPath
  $performance=Get-PerformanceSettings $testPath
  if($performance.trimPaddedText){throw 'Padded text optimization must default off'}
  if($performance.vaeEncodeDevice -ne 'inherit'){throw 'Encode placement must inherit by default'}
  if($performance.vaePseudoDevice -ne 'inherit'){throw 'Pseudo placement must inherit by default'}
  if(-not $performance.srGraph -or -not $performance.sharedVaeStream -or -not $performance.compileVae -or $performance.fixedPromptGraph -or $performance.vaeDevice -ne 'output' -or $performance.rifeDevice -ne 'model'){throw 'Performance settings were not retained'}
  '{"vaeDevice":"model","vaeEncodeDevice":"output","compactPromptGraphs":true,"trimPaddedText":true}' | Set-Content -LiteralPath $testPath
  $split=Get-PerformanceSettings $testPath
  if($split.vaeDevice -ne 'model' -or $split.vaeEncodeDevice -ne 'output'){throw 'Split encode placement was not retained'}
  if(-not $split.compactPromptGraphs){throw 'Compact graph selection was not retained'}
  if(-not $split.trimPaddedText){throw 'Padded text optimization was not retained'}
  '{"vaeDevice":"model","vaeEncodeDevice":"model","vaePseudoDevice":"output"}' | Set-Content -LiteralPath $testPath
  $pseudo=Get-PerformanceSettings $testPath
  if($pseudo.vaeDevice -ne 'model' -or $pseudo.vaeEncodeDevice -ne 'model' -or $pseudo.vaePseudoDevice -ne 'output'){throw 'Independent pseudo placement was not retained'}
  foreach($invalid in @('{"trimPaddedText":"false"}','{"srGraph":"false"}','{"sharedVaeStream":"false"}','{"compileVae":"false"}','{"vaeDevice":"cuda:7"}','{"vaeEncodeDevice":"cuda:7"}','{"vaePseudoDevice":"cuda:7"}','{"unknown":true}')){
    $invalid | Set-Content -LiteralPath $testPath
    $rejected=$false
    try{Get-PerformanceSettings $testPath | Out-Null}catch{$rejected=$true}
    if(-not $rejected){throw 'Invalid performance settings should be rejected'}
  }
  Write-Output 'Launcher parsing and setting checks passed'
} finally { if(Test-Path -LiteralPath $testPath){Remove-Item -LiteralPath $testPath} }
