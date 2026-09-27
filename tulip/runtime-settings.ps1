# Read at every launch, including SDK/supervisor restarts.
function Get-ResidentTextSetting([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) { return $false }
  $settings = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
  if ($settings.residentText -isnot [bool]) {
    throw 'runtime-settings.json residentText must be a JSON boolean'
  }
  return [bool]$settings.residentText
}

function Get-PerformanceSettings([string]$Path) {
  $result = @{ compileVae=$false; sharedVaeStream=$false; srGraph=$false; fixedPromptGraph=$true; compactPromptGraphs=$false; singlePromptGraph=$false; trimPaddedText=$false; vaeDevice='model'; vaeEncodeDevice='inherit'; vaePseudoDevice='inherit'; rifeDevice='output' }
  if (-not (Test-Path -LiteralPath $Path)) { return $result }
  $settings = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
  foreach ($property in $settings.PSObject.Properties) {
    if (-not $result.ContainsKey($property.Name)) { throw "Unknown performance setting: $($property.Name)" }
    if ($property.Name -in @('compileVae','fixedPromptGraph','sharedVaeStream','srGraph','compactPromptGraphs','singlePromptGraph','trimPaddedText')) {
      if ($property.Value -isnot [bool]) { throw "$($property.Name) must be a JSON boolean" }
    } elseif ($property.Name -in @('vaeEncodeDevice','vaePseudoDevice')) {
      if ($property.Value -notin @('inherit','model','output')) { throw "$($property.Name) must be inherit, model or output" }
    } elseif ($property.Value -notin @('model','output')) {
      throw "$($property.Name) must be model or output"
    }
    $result[$property.Name]=$property.Value
  }
  return $result
}
