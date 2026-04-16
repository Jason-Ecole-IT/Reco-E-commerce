#!/bin/bash
# Script de démarrage Docker pour le système de recommandation

echo "=========================================="
echo "🐳 Démarrage Docker Compose"
echo "=========================================="

# Vérifier Docker
echo "📋 Vérification Docker..."
docker --version
docker-compose --version

# Créer les répertoires nécessaires
echo "📁 Création des répertoires..."
mkdir -p data/processed
mkdir -p models
mkdir -p logs
mkdir -p mlflow-artifacts
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/grafana/datasources

# Construire et démarrer les services
echo "🚀 Construction et démarrage des services..."
docker-compose up -d --build

# Attendre que les services démarrent
echo "⏳ Attente démarrage des services (10s)..."
sleep 10

# Vérifier les services
echo "🔍 Vérification des services..."
docker-compose ps

echo ""
echo "=========================================="
echo "✅ Services Docker démarrés !"
echo "=========================================="
echo "API: http://localhost:8000"
echo "Redis: localhost:6379"
echo "Prometheus: http://localhost:9090"
echo "Grafana: http://localhost:3000 (admin/admin)"
echo "MLflow: http://localhost:5000"
echo ""
echo "Logs: docker-compose logs -f"
echo "Arrêter: docker-compose down"
echo "=========================================="
