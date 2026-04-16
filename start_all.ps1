# Script de demarrage systeme de recommandation (Windows PowerShell)

Write-Host "==========================================" -ForegroundColor Green
Write-Host "Demarrage Systeme de Recommandation" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

# Verifier Python
Write-Host "Verification Python..." -ForegroundColor Cyan
python --version

# Creer les repertoires necessaires
Write-Host "Creation des repertoires..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path data\processed | Out-Null
New-Item -ItemType Directory -Force -Path models | Out-Null
New-Item -ItemType Directory -Force -Path logs | Out-Null
New-Item -ItemType Directory -Force -Path mlflow-artifacts | Out-Null

# Installer les dependances
Write-Host "Installation des dependances..." -ForegroundColor Cyan
pip install -r requirements.txt

# Demarrer l'API
Write-Host "Demarrage API de recommandation..." -ForegroundColor Cyan
Start-Process python -ArgumentList "src/serving/recommendation_api_enhanced.py" -NoNewWindow

# Attendre que l'API demarre
Write-Host "Attente demarrage API (5s)..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Verifier l'API
Write-Host "Verification API..." -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing
    Write-Host "API operationnelle" -ForegroundColor Green
} catch {
    Write-Host "API non disponible" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Systeme demarre avec succes !" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host "API: http://localhost:8000" -ForegroundColor White
Write-Host "Health: http://localhost:8000/health" -ForegroundColor White
Write-Host "Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "Pour arreter: Ctrl+C dans le terminal" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Green
