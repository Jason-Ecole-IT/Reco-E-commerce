#!/bin/bash
# Script de démarrage complet pour le système de recommandation

echo "=========================================="
echo "🚀 Démarrage Système de Recommandation"
echo "=========================================="

# Vérifier Python
echo "📋 Vérification Python..."
python --version || python3 --version

# Créer les répertoires nécessaires
echo "📁 Création des répertoires..."
mkdir -p data/processed
mkdir -p models
mkdir -p logs
mkdir -p mlflow-artifacts

# Installer les dépendances
echo "📦 Installation des dépendances..."
pip install -r requirements.txt

# Démarrer Redis (si installé)
if command -v redis-server &> /dev/null; then
    echo "🔴 Démarrage Redis..."
    redis-server --daemonize yes
else
    echo "⚠️  Redis non installé, utilisation fallback"
fi

# Démarrer l'API
echo "🌐 Démarrage API de recommandation..."
python src/serving/recommendation_api_enhanced.py &
API_PID=$!

# Attendre que l'API démarre
echo "⏳ Attente démarrage API (5s)..."
sleep 5

# Vérifier l'API
echo "🔍 Vérification API..."
curl -s http://localhost:8000/health || echo "⚠️  API non disponible"

echo ""
echo "=========================================="
echo "✅ Système démarré avec succès !"
echo "=========================================="
echo "API: http://localhost:8000"
echo "Health: http://localhost:8000/health"
echo "Docs: http://localhost:8000/docs"
echo ""
echo "Pour arrêter: kill $API_PID"
echo "=========================================="
