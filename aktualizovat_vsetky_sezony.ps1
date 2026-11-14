$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = ".\.venv\Scripts\python.exe"

Write-Host "1/4 Stahujem vsetky sezony Premier League (1993/94 - 2026/27)..." -ForegroundColor Cyan
& $py -m src.download

Write-Host "`n2/4 Cistim a zlucujem zapasy..." -ForegroundColor Cyan
& $py -m src.prepare

Write-Host "`n3/4 Pocitam Elo, Glicko a Glicko-2 a generujem grafy..." -ForegroundColor Cyan
& $py -m src.evaluate_ratings

Write-Host "`n4/4 Aktualizujem Word dokument prace SOC..." -ForegroundColor Cyan
& $py tools\build_praca_ratings.py

Write-Host "`nHotovo! Teraz mozes spustit aplikaciu:" -ForegroundColor Green
Write-Host ".\.venv\Scripts\streamlit.exe run streamlit_app.py" -ForegroundColor Yellow
