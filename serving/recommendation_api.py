"""
API de recommandation avec FastAPI
Model serving et endpoints de prédiction
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
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
from functools import lru_cache

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialiser FastAPI
app = FastAPI(
    title="E-commerce Recommendation API",
    description="API de recommandation pour système e-commerce",
    version="1.0.0"
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration Redis
try:
    redis_client = redis.Redis(
        host='localhost',
        port=6379,
        db=0,
        decode_responses=True
    )
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

# Chargement des modèles
class ModelLoader:
    """Chargeur de modèles pour le serving"""
    
    def __init__(self):
        self.models = {}
        self.feature_stores = {}
        
    def load_model(self, model_path: str, model_name: str):
        """Charger un modèle depuis le disque"""
        try:
            if os.path.exists(f"{model_path}/{model_name}.pkl"):
                model = joblib.load(f"{model_path}/{model_name}.pkl")
                self.models[model_name] = model
                logger.info(f"Modèle {model_name} chargé")
                return model
            else:
                logger.warning(f"Modèle {model_name} non trouvé")
                return None
        except Exception as e:
            logger.error(f"Erreur chargement modèle {model_name}: {e}")
            return None
    
    def load_feature_store(self, feature_store_path: str, category: str):
        """Charger le feature store"""
        try:
            df = pd.read_csv(feature_store_path)
            self.feature_stores[category] = df
            logger.info(f"Feature store {category} chargé: {len(df)} enregistrements")
            return df
        except Exception as e:
            logger.error(f"Erreur chargement feature store {category}: {e}")
            return None
    
    def get_model(self, model_name: str):
        """Récupérer un modèle"""
        return self.models.get(model_name)
    
    def get_feature_store(self, category: str):
        """Récupérer le feature store"""
        return self.feature_stores.get(category)

# Initialiser le chargeur de modèles
model_loader = ModelLoader()

# Charger les modèles au démarrage
@app.on_event("startup")
async def startup_event():
    """Charger les modèles au démarrage"""
    logger.info("Chargement des modèles au démarrage")
    
    # Charger le modèle de recommandation
    model_path = "data/output"
    model_loader.load_model(model_path, "collaborative_filtering_model")
    
    # Charger les feature stores
    model_loader.load_feature_store("data/clean_data/features/clothing_features_feature_store.csv", "clothing")
    model_loader.load_feature_store("data/clean_data/features/electronics_features_feature_store.csv", "electronics")
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
    
    logger.info("Démarrage de l'API terminé")

# Fonctions de cache Redis
def cache_get(key: str):
    """Récupérer depuis le cache Redis"""
    try:
        if redis_client:
            value = redis_client.get(key)
            if value:
                return json.loads(value)
        return None
    except Exception as e:
        logger.warning(f"Erreur cache get: {e}")
        return None

def cache_set(key: str, value: dict, ttl: int = 3600):
    """Stocker dans le cache Redis"""
    try:
        if redis_client:
            redis_client.setex(key, ttl, json.dumps(value))
    except Exception as e:
        logger.warning(f"Erreur cache set: {e}")

def generate_recommendations(user_id: str, top_n: int, model_data: dict) -> list:
    """Générer des recommandations pour un utilisateur"""
    try:
        user_factors = model_data['user_factors']
        item_factors = model_data['item_factors']
        user_ids = model_data['user_ids']
        item_ids = model_data['item_ids']
        item_similarity = model_data['item_similarity']
        
        if user_id not in user_ids:
            # Cold start: retourner les produits les plus populaires
            popular_items = (
                model_data['user_item_matrix']
                .sum(axis=0)
                .sort_values(ascending=False)
                .head(top_n)
            )
            return [
                {
                    "product_id": item_ids[idx],
                    "score": float(score),
                    "reason": "popular_fallback"
                }
                for idx, score in popular_items.items()
            ]
        
        # Trouver l'index de l'utilisateur
        user_idx = user_ids.index(user_id)
        user_vector = user_factors[user_idx]
        
        # Calculer les scores de similarité
        scores = {}
        for item_idx, item_id in enumerate(item_ids):
            item_vector = item_factors[item_idx]
            score = np.dot(user_vector, item_vector)
            scores[item_id] = score
        
        # Trier et retourner les top N
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {
                "product_id": product_id,
                "score": float(score),
                "reason": "collaborative_filtering"
            }
            for product_id, score in sorted_scores[:top_n]
        ]
        
    except Exception as e:
        logger.error(f"Erreur génération recommandations: {e}")
        return []

# Endpoints de l'API
@app.get("/")
async def root():
    """Racine de l'API"""
    return {
        "message": "E-commerce Recommendation API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Vérification de santé"""
    return {
        "status": "healthy",
        "models_loaded": len(model_loader.models),
        "feature_stores_loaded": len(model_loader.feature_stores),
        "redis_available": redis_client is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    """Obtenir des recommandations pour un utilisateur"""
    try:
        # Vérifier le cache
        cache_key = f"recommendations:{request.user_id}:{request.category}:{request.num_recommendations}"
        cached_result = cache_get(cache_key)
        
        if cached_result:
            logger.info(f"Cache hit pour {cache_key}")
            return cached_result
        
        # Récupérer le modèle
        model = model_loader.get_model("collaborative_filtering_model")
        if not model:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Générer les recommandations
        recommendations = generate_recommendations(
            request.user_id, 
            request.num_recommendations, 
            model
        )
        
        result = {
            "user_id": request.user_id,
            "category": request.category,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
        # Mettre en cache
        cache_set(cache_key, result)
        
        logger.info(f"Recommandations générées pour {request.user_id}")
        return result
        
        result = {
            "user_id": request.user_id,
            "category": request.category,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
        # Mettre en cache
        cache_set(cache_key, result)
        
        logger.info(f"Recommandations générées pour {request.user_id}")
        return result
        
    except Exception as e:
        logger.error(f"Erreur génération recommandations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_rating")
async def predict_rating(request: RatingPredictionRequest):
    """Prédire la note d'un utilisateur pour un produit"""
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
                "timestamp": datetime.now().isoformat()
            }
        
        # Mettre en cache
        cache_set(cache_key, result)
        
        logger.info(f"Prédiction générée pour {request.user_id} / {request.product_id}")
        return result
        
    except Exception as e:
        logger.error(f"Erreur prédiction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_recommend")
async def batch_recommend(request: BatchRecommendationRequest):
    """Recommandations batch pour plusieurs utilisateurs"""
    try:
        results = []
        
        for user_id in request.user_ids:
            # Créer une requête individuelle
            single_request = RecommendationRequest(
                user_id=user_id,
                category=request.category,
                num_recommendations=request.num_recommendations
            )
            
            # Obtenir les recommandations
            try:
                recommendations = await get_recommendations(single_request)
                results.append({
                    "user_id": user_id,
                    "status": "success",
                    "recommendations": recommendations
                })
            except Exception as e:
                results.append({
                    "user_id": user_id,
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "category": request.category,
            "num_users": len(request.user_ids),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur batch recommandations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Statistiques de l'API"""
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
            "redis_available": redis_client is not None,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/products/{category}")
async def get_products(category: str, limit: int = 100):
    """Lister les produits d'une catégorie"""
    try:
        feature_store = model_loader.get_feature_store(category)
        if not feature_store:
            raise HTTPException(status_code=404, detail=f"Category {category} not found")
        
        products = feature_store['asin'].unique()[:limit]
        
        return {
            "category": category,
            "products": list(products),
            "count": len(products),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur get products: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cache")
async def clear_cache():
    """Vider le cache Redis"""
    try:
        if redis_client:
            redis_client.flushdb()
            return {"status": "success", "message": "Cache cleared"}
        else:
            return {"status": "error", "message": "Redis not available"}
    except Exception as e:
        logger.error(f"Erreur clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
