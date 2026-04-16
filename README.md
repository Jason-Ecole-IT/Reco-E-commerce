# Reco E-commerce

## Projet M1 SPAR - Moteur de recommandation e-commerce

### Description

Ce projet développe un moteur de recommandation e-commerce basé sur PySpark ALS, des embeddings TensorFlow et Feast comme feature store. L’objectif est de proposer des recommandations personnalisées, efficaces et adaptées à un environnement e-commerce.

### Objectifs

- Améliorer le taux de clics (CTR) d’au moins 15% par rapport à une recommandation aléatoire
- Atteindre une latence API cible inférieure à 50ms
- Supporter un volume important d’utilisateurs et de produits
- Livrer un prototype documenté et déployable

### Stack technique

- Big Data : PySpark 3.4, Delta Lake 2.4
- ML : PySpark ALS, TensorFlow 2.14, Scikit-learn
- Feature Store : Feast 0.38
- Serving : FastAPI, Redis, Streamlit
- MLOps : MLflow 2.8, Docker
- Monitoring : Prometheus, Grafana

## Structure du projet

- `api/` : API backend et services de recommandation
- `app/` : interface utilisateur et dashboard
- `data/` : données brutes, données nettoyées et rapports
- `docs/` : documentation du projet
- `docker/` : conteneurs et orchestration
- `etl/` : scripts d’ingestion, de nettoyage et d’analyse
- `features/` : logique de feature engineering
- `ml/` : modèles et entraînement
- `notebooks/` : exploration et prototypes

## Démarrage rapide

1. Installer les dépendances :

```bash
pip install -r requirements.txt
```

2. Télécharger les données :

```bash
python etl/download_data.py
```

3. Nettoyer les données :

```bash
python etl/data_cleaning.py
```

4. Analyser la qualité des données :

```bash
python etl/data_quality_analysis.py
```

5. Régénérer le modèle:

```bash
python ml/train_recommender_sklearn.py
```

6. Lancer l'API:

```bash
uvicorn serving.recommendation_api:app --host 0.0.0.0 --port 8000 --reload
```

7. Lancer le dashboard:

```bash
streamlit run app/dashboard.py```

8. Consulter la documentation :

- `docs/architecture.md`
- `docs/architecture_technique.md`
- `docs/kpis_metrics.md`
- `docs/project_charter.md`
- `docs/planning_5jours.md`
- `docs/management-artifact.md`

## Données principales

- `data/clean_data/amazon_reviews_clothing_clean.json`
- `data/clean_data/amazon_reviews_electronics_clean.json`
- `data/report/quality_report_clothing.json`
- `data/report/quality_report_electronics.json`

## Dataset

Le projet utilise des datasets basée sur les review Amazon, accessibles via ce lien: [Amazon Reviews](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/), nous avons sélectionné le dataset 5-core, qui est un sous-ensemble filtrés pour ne conserver que les utilisateurs et produits avec au moins 5 interactions, afin de garantir une qualité suffisant pour l'entrainement de notre moteur de recommandation.

Sur ce dataset nous avons utilisé un autre sous-ensemble, qui contient uniquement les catégories "Clothing" et "Electronics" à des fin de test et de démonstration.

## Notes

Ce projet est conçu pour être extensible et reproductible : les pipelines ETL sont séparés des modèles ML, et l’architecture permet d’ajouter facilement un service de recommandation temps réel.
