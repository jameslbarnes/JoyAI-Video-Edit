param([Parameter(Mandatory=$true)][ValidateSet('on','off')][string]$Mode)
$ErrorActionPreference='Stop'
$path=Join-Path $PSScriptRoot 'neural-rendering.json'
$config=Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
$config.enabled=($Mode -eq 'on')
$temp=Join-Path $PSScriptRoot ('neural-rendering-'+[guid]::NewGuid().ToString('N')+'.tmp')
$backup=Join-Path $PSScriptRoot ('neural-rendering-'+(Get-Date -Format 'yyyyMMdd-HHmmss')+'.bak')
$config | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $temp -Encoding UTF8
[System.IO.File]::Replace($temp,$path,$backup)
Write-Output "Neural Rendering requested: $Mode (takes effect at the next output chunk)."
