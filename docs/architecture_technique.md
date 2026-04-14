# Architecture Technique - Moteur de Recommandation E-commerce

## Vue d'Ensemble

Ce document décrit l'architecture technique complète du moteur de recommandation e-commerce utilisant PySpark ALS, TensorFlow embeddings et Feast pour le feature store.

## Architecture End-to-End

```
                    +---------------------+
                    |   Data Sources      |
                    +---------------------+
                    | Amazon Reviews v2   |
                    | Product Metadata    |
                    | User Interactions  |
                    +----------+----------+
                               |
                    +----------v----------+
                    |   Data Ingestion   |
                    +----------+----------+
                    | PySpark ETL        |
                    | Delta Lake         |
                    | Data Validation    |
                    +----------+----------+
                               |
                    +----------v----------+
                    | Feature Engineering|
                    +----------+----------+
                    | Feast Feature Store|
                    | User/Item Features |
                    | Temporal Features  |
                    +----------+----------+
                               |
                    +----------v----------+
                    |   Model Training   |
                    +----------+----------+
                    | PySpark ALS        |
                    | TensorFlow Embed   |
                    | MLflow Tracking    |
                    +----------+----------+
                               |
                    +----------v----------+
                    |   Model Serving     |
                    +----------+----------+
                    | FastAPI REST API   |
                    | Redis Cache        |
                    | Load Balancer      |
                    +----------+----------+
                               |
                    +----------v----------+
                    |   User Interface   |
                    +----------+----------+
                    | Streamlit Dashboard|
                    | A/B Testing UI     |
                    | Analytics          |
                    +---------------------+
```

## Composants Techniques

### 1. Data Layer
- **Source**: Amazon Reviews v2 (Electronics + Clothing)
- **Format**: JSON Lines compressé (.json.gz)
- **Volume**: ~1.5GB (Electronics) + ~150MB (Clothing)
- **Storage**: Delta Lake pour ACID transactions
- **Validation**: Great Expectations pour la qualité

### 2. Processing Layer
- **Framework**: Apache Spark 3.4+ (PySpark)
- **Orchestration**: MLflow Pipelines
- **Feature Store**: Feast 0.38
- **Cache**: Redis 7.0
- **Monitoring**: Prometheus + Grafana

### 3. Model Layer
- **Collaborative Filtering**: PySpark ALS
- **Deep Learning**: TensorFlow 2.14 embeddings
- **Hybrid Approach**: Combinaison ALS + Content-based
- **Evaluation**: RMSE, Precision@K, NDCG

### 4. Serving Layer
- **API**: FastAPI avec OpenAPI docs
- **Real-time**: <50ms latence cible
- **Cache**: Redis pour recommandations
- **Scaling**: Horizontal avec load balancing

### 5. Interface Layer
- **Dashboard**: Streamlit pour analytics
- **A/B Testing**: Interface de test
- **Monitoring**: Métriques temps réel

## Flux de Données

### Pipeline ETL (Batch)
1. **Ingestion**: Chargement des données brutes
2. **Cleaning**: Validation et nettoyage
3. **Transformation**: Feature engineering
4. **Storage**: Sauvegarde dans Delta Lake

### Pipeline ML (Training)
1. **Feature Extraction**: User/Item features
2. **Model Training**: ALS + TensorFlow
3. **Evaluation**: Cross-validation
4. **Registry**: MLflow Model Registry

### Pipeline Serving (Real-time)
1. **Request**: API endpoint appel
2. **Cache Check**: Redis lookup
3. **Prediction**: Model inference
4. **Response**: Recommandations retournées

## Challenges Techniques Identifiés

### 1. Cold Start Problem
- **Users**: 77% avec 1 review (Electronics)
- **Products**: Distribution inégale
- **Solution**: Content-based + popularity fallback

### 2. Scalabilité
- **Volume**: 1.6M+ reviews
- **Users**: 36K+ unique users
- **Products**: 5K+ unique products
- **Solution**: Spark distributed computing

### 3. Performance
- **Latence**: <50ms cible
- **Throughput**: 10K+ req/s
- **Solution**: Redis cache + model optimization

### 4. Data Quality
- **Missing values**: reviewerName (0.6%)
- **Bias**: Distribution notes (61% notes 4-5)
- **Solution**: Data validation + normalization

## Stack Technique Détaillé

### BigData & Processing
```yaml
Spark:
  version: 3.4+
  features: Adaptive Query Execution, Kryo Serialization
  
Delta Lake:
  version: 2.4+
  features: ACID transactions, Time Travel
  
Feast:
  version: 0.38
  features: Feature store, Online serving
```

### Machine Learning
```yaml
PySpark ALS:
  algorithm: Alternating Least Squares
  parameters: rank=50, regParam=0.1
  
TensorFlow:
  version: 2.14
  architecture: Embedding layers
  training: Distributed training
  
MLflow:
  version: 2.8
  features: Tracking, Registry, Serving
```

### APIs & Serving
```yaml
FastAPI:
  version: 0.104
  features: Async, OpenAPI docs
  
Redis:
  version: 7.0
  features: In-memory cache, Pub/Sub
  
Streamlit:
  version: 1.28
  features: Real-time dashboard, Components
```

## Monitoring & Observabilité

### Métriques Business
- **CTR**: Click-Through Rate
- **Conversion**: Taux de conversion
- **Revenue**: Revenue per user
- **Diversity**: Diversité des recommandations

### Métriques Techniques
- **Latence**: API response time
- **Throughput**: Requests per second
- **Cache Hit Rate**: Efficacité du cache
- **Model Accuracy**: RMSE, Precision@K

### Infrastructure Monitoring
- **Spark**: Cluster metrics, job performance
- **Redis**: Memory usage, hit rate
- **API**: Response times, error rates
- **System**: CPU, memory, disk usage

## Sécurité & Compliance

### Data Privacy
- **Anonymisation**: User IDs hashés
- **Encryption**: TLS pour les APIs
- **Access Control**: RBAC pour les services

### Model Governance
- **Versioning**: MLflow model registry
- **Audit Trail**: Logs structurés
- **Explainability**: Feature importance tracking

## Déploiement & Scalabilité

### Containerisation
- **Docker**: Images pour chaque service
- **Docker Compose**: Orchestration locale
- **Kubernetes**: Production scaling

### CI/CD Pipeline
- **Testing**: Unit tests + integration tests
- **Quality**: Code coverage >80%
- **Deployment**: Automated deployment

### High Availability
- **Redundancy**: Multi-instance deployment
- **Failover**: Automatic failover
- **Backup**: Regular data backups

## Estimation des Ressources

### Cluster Spark (Production)
- **Driver**: 4 cores, 16GB RAM
- **Workers**: 8 cores, 32GB RAM × 3
- **Storage**: 100GB SSD

### Serving Infrastructure
- **API Servers**: 2 cores, 4GB RAM × 2
- **Redis**: 4 cores, 8GB RAM
- **Load Balancer**: 2 cores, 2GB RAM

### Total Estimated Cost
- **Compute**: ~$500/month (cloud)
- **Storage**: ~$100/month
- **Monitoring**: ~$50/month
