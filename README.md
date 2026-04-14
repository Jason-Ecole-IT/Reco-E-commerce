# Moteur de Recommandation E-commerce

**Projet M1SPAR - Python Spark et projets IA avancés**

## 📋 Description

Moteur de recommandation e-commerce utilisant PySpark ALS pour le collaborative filtering, TensorFlow embeddings pour le deep learning, et Feast pour le feature store.

**Objectifs métier :**
- Augmenter CTR de 15% vs recommandations random
- Recommandations personnalisées <50ms latence  
- Support 10M utilisateurs, 1M produits
- A/B testing avec significance statistique

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Source   │    │   Processing     │    │   Serving       │
│                 │    │                  │    │                 │
│ Amazon Reviews  │───▶│ PySpark ALS      │───▶│ FastAPI         │
│ Metadata        │    │ TensorFlow       │    │ Redis Cache     │
└─────────────────┘    │ Feature Store    │    │ Streamlit UI    │
                       └──────────────────┘    └─────────────────┘
```

## 🛠️ Stack Technique

- **BigData**: PySpark 3.4, Delta Lake 2.4
- **ML**: PySpark ALS, TensorFlow 2.14, Scikit-learn
- **Feature Store**: Feast 0.38
- **Serving**: FastAPI, Redis, Streamlit
- **MLOps**: MLflow 2.8, Docker
- **Monitoring**: Prometheus, Grafana

## 📁 Structure du Projet

```
moteur_recommandation/
├── data/                   # Données brutes et traitées
├── notebooks/              # Jupyter notebooks d'exploration
├── src/                    # Code source
│   ├── data/              # Pipeline ETL
│   ├── models/            # Modèles ML
│   ├── features/          # Feature engineering
│   ├── serving/           # API et serving
│   └── utils/             # Utilitaires
├── tests/                  # Tests unitaires et intégration
├── configs/               # Configurations
├── docker/                # Docker files
└── docs/                  # Documentation
```

## 🚀 Quick Start

1. Installation des dépendances :
```bash
pip install -r requirements.txt
```

2. Téléchargement des données :
```bash
python src/data/download_data.py
```

3. Lancement du pipeline :
```bash
python src/main.py
```

## 📊 KPIs et Métriques

**Business KPIs:**
- Click-Through Rate (CTR)
- Conversion Rate
- Revenue per User
- Recommendation Diversity

**Technical Metrics:**
- Latence API (<50ms)
- Throughput (req/s)
- Model Accuracy (RMSE, Precision@k)
- Cache Hit Rate

## 👥 Équipe

- Taille recommandée : 2-3 étudiants
- Rôles : Data Engineer, ML Engineer, Full-stack Developer

## 📅 Planning (5 jours)

- **Jour 1**: Setup et exploration données
- **Jour 2**: Pipeline ETL et feature engineering  
- **Jour 3**: Modélisation ML et entraînement
- **Jour 4**: Déploiement et interfaces
- **Jour 5**: Finalisation et soutenance
