param([string]$Evidence='tulip/evidence/strength-fine-20260911')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$ffmpeg=Join-Path $repo '../easel/external/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe'
Push-Location $repo
try {
    & $ffmpeg -hide_banner -loglevel error -n -i "$Evidence/porcelain-position8/output.mp4" -i "$Evidence/porcelain-blend0.3/output.mp4" -i "$Evidence/porcelain-blend0.5/output.mp4" -filter_complex '[0:v]scale=432:240[a];[1:v]scale=432:240[b];[2:v]scale=432:240[c];[a][b][c]hstack=inputs=3[v]' -map '[v]' -an -c:v libx264 -crf 18 -pix_fmt yuv420p "$Evidence/porcelain-comparison.mp4"
    if($LASTEXITCODE -ne 0){throw 'Comparison encoding failed'}
} finally {Pop-Location}
