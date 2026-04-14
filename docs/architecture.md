# Architecture fonctionnelle

Ce document présente l’architecture fonctionnelle du projet. Pour les détails techniques et la description des composants, consultez `docs/architecture_technique.md`.

## Vue d’ensemble

```mermaid
flowchart LR
    A[Sources de données] --> B[Ingestion ETL]
    B --> C[Stockage Delta Lake]
    C --> D[Feature Store Feast]
    D --> E[Entraînement et évaluation]
    E --> F[API FastAPI]
    F --> G[Cache Redis]
    G --> H[Application utilisateur]
```

## Cas d’usage

```mermaid
flowchart TD
    U[Utilisateur] -->|clic / historique| R[Requête de recommandation]
    R --> S[API FastAPI]
    S --> T{Cache Redis}
    T -->|hit| U
    T -->|miss| V[Modèle ML]
    V --> W[Recommandations]
    W --> U
```

## Description des flux

- **Ingestion** : collecte et nettoyage des données Amazon Reviews
- **Stockage** : persistance des tables dans Delta Lake
- **Feature Store** : publication des features utilisateur/produit avec Feast
- **Modélisation** : entraînement hybride ALS + embeddings
- **Serving** : API FastAPI avec cache Redis
- **Interface** : dashboard Streamlit et interface utilisateur
