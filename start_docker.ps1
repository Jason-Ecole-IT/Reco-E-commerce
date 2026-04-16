# Script de démarrage Docker pour le système de recommandation (Windows PowerShell)

Write-Host "==========================================" -ForegroundColor Green
Write-Host "🐳 Démarrage Docker Compose" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

# Vérifier Docker
Write-Host "📋 Vérification Docker..." -ForegroundColor Cyan
docker --version
docker-compose --version

# Créer les répertoires nécessaires
Write-Host "📁 Création des répertoires..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path data\processed
New-Item -ItemType Directory -Force -Path models
New-Item -ItemType Directory -Force -Path logs
New-Item -ItemType Directory -Force -Path mlflow-artifacts
New-Item -ItemType Directory -Force -Path monitoring\grafana\dashboards
New-Item -ItemType Directory -Force -Path monitoring\grafana\datasources

# Construire et démarrer les services
Write-Host "🚀 Construction et démarrage des services..." -ForegroundColor Cyan
docker-compose up -d --build

# Attendre que les services démarrent
Write-Host "⏳ Attente démarrage des services (10s)..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Vérifier les services
Write-Host "🔍 Vérification des services..." -ForegroundColor Cyan
docker-compose ps

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "✅ Services Docker démarrés !" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host "API: http://localhost:8000" -ForegroundColor White
Write-Host "Redis: localhost:6379" -ForegroundColor White
Write-Host "Prometheus: http://localhost:9090" -ForegroundColor White
Write-Host "Grafana: http://localhost:3000 (admin/admin)" -ForegroundColor White
Write-Host "MLflow: http://localhost:5000" -ForegroundColor White
Write-Host ""
Write-Host "Logs: docker-compose logs -f" -ForegroundColor Yellow
Write-Host "Arrêter: docker-compose down" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Green
