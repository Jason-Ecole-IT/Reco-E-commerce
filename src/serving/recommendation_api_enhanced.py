"""
API de recommandation améliorée avec métriques Prometheus et monitoring
Version enhanced avec observabilité complète
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import joblib
import json
import logging
from datetime import datetime
import redis
import os
import time
from functools import lru_cache

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialiser FastAPI
app = FastAPI(
    title="E-commerce Recommendation API Enhanced",
    description="API de recommandation avec monitoring et métriques Prometheus",
    version="2.0.0"
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrumentation Prometheus
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app, endpoint="/metrics")

# Configuration Redis
try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=0,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5
    )
    redis_client.ping()
    logger.info("Redis client initialisé")
except Exception as e:
    logger.warning(f"Redis non disponible: {e}")
    redis_client = None

# Modèles Pydantic
class RecommendationRequest(BaseModel):
    user_id: str
    category: Optional[str] = "electronics"
    num_recommendations: Optional[int] = 10

class RatingPredictionRequest(BaseModel):
    user_id: str
    product_id: str
    category: Optional[str] = "electronics"

class BatchRecommendationRequest(BaseModel):
    user_ids: List[str]
    category: Optional[str] = "electronics"
    num_recommendations: Optional[int] = 10

# Métriques personnalisées
request_count = 0
request_latency = []

# Chargement des modèles
class ModelLoader:
    """Chargeur de modèles avec monitoring"""
    
    def __init__(self):
        self.models = {}
        self.feature_stores = {}
        self.load_times = {}
        
    def load_model(self, model_path: str, model_name: str):
        """Charger un modèle avec monitoring"""
        try:
            start_time = time.time()
            
            if os.path.exists(f"{model_path}/{model_name}.pkl"):
                model = joblib.load(f"{model_path}/{model_name}.pkl")
                self.models[model_name] = model
                
                load_time = time.time() - start_time
                self.load_times[model_name] = load_time
                
                logger.info(f"Modèle {model_name} chargé en {load_time:.2f}s")
                return model
            else:
                logger.warning(f"Modèle {model_name} non trouvé")
                return None
        except Exception as e:
            logger.error(f"Erreur chargement modèle {model_name}: {e}")
            return None
    
    def load_feature_store(self, feature_store_path: str, category: str):
        """Charger le feature store avec monitoring"""
        try:
            start_time = time.time()
            
            df = pd.read_csv(feature_store_path)
            self.feature_stores[category] = df
            
            load_time = time.time() - start_time
            logger.info(f"Feature store {category} chargé en {load_time:.2f}s: {len(df)} enregistrements")
            return df
        except Exception as e:
            logger.error(f"Erreur chargement feature store {category}: {e}")
            return None
    
    def get_feature_store(self, category: str):
        """Récupérer le feature store pour une catégorie"""
        return self.feature_stores.get(category)

# Initialiser le chargeur de modèles
model_loader = ModelLoader()

# Charger les modèles au démarrage
@app.on_event("startup")
async def startup_event():
    """Charger les modèles au démarrage avec monitoring"""
    logger.info("Démarrage de l'API avec monitoring")
    start_time = time.time()
    
    # Charger les feature stores
    model_loader.load_feature_store(
        "data/processed/electronics_features_feature_store.csv",
        "electronics"
    )
    model_loader.load_feature_store(
        "data/processed/clothing_features_feature_store.csv",
        "clothing"
    )
    
    # Charger les modèles s'ils existent
    model_loader.load_model("models/ml_electronics", "logistic_regression")
    model_loader.load_model("models/ml_electronics", "random_forest")
    model_loader.load_model("models/ml_electronics", "gradient_boosting")
    
    model_loader.load_model("models/ml_clothing", "logistic_regression")
    model_loader.load_model("models/ml_clothing", "random_forest")
    model_loader.load_model("models/ml_clothing", "gradient_boosting")
    
    startup_time = time.time() - start_time
    logger.info(f"Démarrage terminé en {startup_time:.2f}s")

# Middleware pour le tracking des requêtes
@app.middleware("http")
async def track_requests(request, call_next):
    """Tracker les requêtes avec métriques"""
    global request_count, request_latency
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    request_count += 1
    request_latency.append(process_time)
    
    # Garder seulement les 1000 dernières valeurs
    if len(request_latency) > 1000:
        request_latency = request_latency[-1000:]
    
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-Count"] = str(request_count)
    
    return response

# Fonctions de cache Redis
def cache_get(key: str):
    """Récupérer depuis le cache Redis"""
    try:
        if redis_client:
            start_time = time.time()
            value = redis_client.get(key)
            if value:
                logger.debug(f"Cache hit pour {key} (temps: {time.time() - start_time:.3f}s)")
                return json.loads(value)
        return None
    except Exception as e:
        logger.warning(f"Erreur cache get: {e}")
        return None

def cache_set(key: str, value: dict, ttl: int = 3600):
    """Stocker dans le cache Redis"""
    try:
        if redis_client:
            start_time = time.time()
            redis_client.setex(key, ttl, json.dumps(value))
            logger.debug(f"Cache set pour {key} (temps: {time.time() - start_time:.3f}s)")
    except Exception as e:
        logger.warning(f"Erreur cache set: {e}")

# Endpoints de l'API
@app.get("/")
async def root():
    """Racine de l'API avec métriques"""
    return {
        "message": "E-commerce Recommendation API Enhanced",
        "version": "2.0.0",
        "status": "running",
        "request_count": request_count,
        "avg_latency_ms": np.mean(request_latency) * 1000 if request_latency else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Vérification de santé détaillée"""
    redis_status = "healthy" if redis_client else "unavailable"
    
    try:
        if redis_client:
            redis_client.ping()
    except:
        redis_status = "unhealthy"
    
    return {
        "status": "healthy",
        "models_loaded": len(model_loader.models),
        "feature_stores_loaded": len(model_loader.feature_stores),
        "redis_status": redis_status,
        "request_count": request_count,
        "avg_latency_ms": np.mean(request_latency) * 1000 if request_latency else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics/custom")
async def custom_metrics():
    """Métriques personnalisées"""
    return {
        "request_count": request_count,
        "avg_latency_ms": np.mean(request_latency) * 1000 if request_latency else 0,
        "p95_latency_ms": np.percentile(request_latency, 95) * 1000 if request_latency else 0,
        "p99_latency_ms": np.percentile(request_latency, 99) * 1000 if request_latency else 0,
        "models_loaded": len(model_loader.models),
        "feature_stores_loaded": len(model_loader.feature_stores),
        "cache_status": "available" if redis_client else "unavailable",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    """Obtenir des recommandations avec monitoring"""
    start_time = time.time()
    
    try:
        # Vérifier le cache
        cache_key = f"recommendations:{request.user_id}:{request.category}:{request.num_recommendations}"
        cached_result = cache_get(cache_key)
        
        if cached_result:
            cache_hit_time = time.time() - start_time
            logger.info(f"Cache hit pour {cache_key} (temps: {cache_hit_time:.3f}s)")
            return cached_result
        
        # Récupérer le feature store
        feature_store = model_loader.get_feature_store(request.category)
        if feature_store is None or feature_store.empty:
            raise HTTPException(status_code=404, detail=f"Category {request.category} not found")
        
        # Filtrer les produits
        products = feature_store['asin'].unique()
        
        # Simulation de recommandations (basée sur popularité)
        product_popularity = feature_store.groupby('asin')['overall'].agg(['count', 'mean'])
        product_popularity = product_popularity.sort_values(['count', 'mean'], ascending=False)
        
        recommendations = []
        for product_id in product_popularity.head(request.num_recommendations).index:
            recommendations.append({
                "product_id": product_id,
                "score": float(product_popularity.loc[product_id, 'count']),
                "avg_rating": float(product_popularity.loc[product_id, 'mean'])
            })
        
        result = {
            "user_id": request.user_id,
            "category": request.category,
            "recommendations": recommendations,
            "cache_hit": False,
            "processing_time_ms": (time.time() - start_time) * 1000,
            "timestamp": datetime.now().isoformat()
        }
        
        # Mettre en cache
        cache_set(cache_key, result)
        
        logger.info(f"Recommandations générées pour {request.user_id} (temps: {(time.time() - start_time) * 1000:.2f}ms)")
        return result
        
    except Exception as e:
        logger.error(f"Erreur génération recommandations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_rating")
async def predict_rating(request: RatingPredictionRequest):
    """Prédire la note avec monitoring"""
    start_time = time.time()
    
    try:
        # Vérifier le cache
        cache_key = f"rating:{request.user_id}:{request.product_id}:{request.category}"
        cached_result = cache_get(cache_key)
        
        if cached_result:
            logger.info(f"Cache hit pour {cache_key}")
            return cached_result
        
        # Récupérer le feature store
        feature_store = model_loader.get_feature_store(request.category)
        if not feature_store:
            raise HTTPException(status_code=404, detail=f"Category {request.category} not found")
        
        # Trouver des données similaires
        product_data = feature_store[feature_store['asin'] == request.product_id]
        
        if len(product_data) == 0:
            # Produit non trouvé, retourner la moyenne globale
            global_avg = feature_store['overall'].mean()
            result = {
                "user_id": request.user_id,
                "product_id": request.product_id,
                "predicted_rating": float(global_avg),
                "confidence": 0.5,
                "method": "global_average",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Utiliser la moyenne du produit
            product_avg = product_data['overall'].mean()
            result = {
                "user_id": request.user_id,
                "product_id": request.product_id,
                "predicted_rating": float(product_avg),
                "confidence": 0.7,
                "method": "product_average",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": datetime.now().isoformat()
            }
        
        # Mettre en cache
        cache_set(cache_key, result)
        
        logger.info(f"Prédiction générée pour {request.user_id} / {request.product_id}")
        return result
        
    except Exception as e:
        logger.error(f"Erreur prédiction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Statistiques détaillées de l'API"""
    try:
        feature_stores_stats = {}
        
        for category, df in model_loader.feature_stores.items():
            feature_stores_stats[category] = {
                "total_reviews": len(df),
                "unique_users": df['reviewerID'].nunique(),
                "unique_products": df['asin'].nunique(),
                "avg_rating": float(df['overall'].mean())
            }
        
        return {
            "feature_stores": feature_stores_stats,
            "models_loaded": list(model_loader.models.keys()),
            "model_load_times": model_loader.load_times,
            "redis_available": redis_client is not None,
            "request_count": request_count,
            "avg_latency_ms": np.mean(request_latency) * 1000 if request_latency else 0,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
