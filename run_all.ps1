[CmdletBinding()]
param(
    [switch]$Force,
    [switch]$SkipTraining
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Chyba virtualne prostredie. Spusti: py -3.12 -m venv .venv a pip install -r requirements.txt"
}

Push-Location $root
try {
    $downloadArgs = @()
    if ($Force) { $downloadArgs += "--force" }

    Write-Host "=== 1/5 Stahovanie dat ===" -ForegroundColor Cyan
    & $python -m src.download @downloadArgs
    if ($LASTEXITCODE -ne 0) { throw "Download zlyhal" }

    Write-Host "=== 2/5 Cistenie dat ===" -ForegroundColor Cyan
    & $python -m src.prepare
    if ($LASTEXITCODE -ne 0) { throw "Prepare zlyhal" }

    Write-Host "=== 3/5 Feature engineering ===" -ForegroundColor Cyan
    & $python -m src.features
    if ($LASTEXITCODE -ne 0) { throw "Features zlyhali" }

    if (-not $SkipTraining) {
        Write-Host "=== 4/5 Trenovanie modelov ===" -ForegroundColor Cyan
        & $python -m src.train
        if ($LASTEXITCODE -ne 0) { throw "Training zlyhal" }

        Write-Host "=== 5/5 Vyhodnotenie ===" -ForegroundColor Cyan
        & $python -m src.evaluate
        if ($LASTEXITCODE -ne 0) { throw "Evaluate zlyhal" }
    }

    Write-Host "Hotovo." -ForegroundColor Green
}
finally {
    Pop-Location
}
