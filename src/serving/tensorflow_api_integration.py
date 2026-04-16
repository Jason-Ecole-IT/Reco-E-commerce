"""
Integration TensorFlow avec l'API de recommandation
Endpoints TensorFlow pour le serving de modèles NCF
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
from datetime import datetime
import sys
import os
import numpy as np

sys.path.append('.')

from src.models.neural_collaborative_filtering import NCFModel

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modèles Pydantic
class TensorFlowRecommendationRequest(BaseModel):
    user_id: str
    category: str = "electronics"
    num_recommendations: int = 10
    model_type: str = "ncf"  # ncf, embeddings

class TensorFlowPredictionRequest(BaseModel):
    user_id: str
    item_id: str
    category: str = "electronics"
    model_type: str = "ncf"

# Gestionnaire de modèles TensorFlow
class TensorFlowModelManager:
    """Gestionnaire pour les modèles TensorFlow"""
    
    def __init__(self):
        """Initialiser le gestionnaire"""
        self.models = {}
        self.model_paths = {
            "ncf_electronics": "models/ncf_trained",
            "ncf_clothing": "models/ncf_trained",
            "embeddings_electronics": "models/embeddings_demo",
            "embeddings_clothing": "models/embeddings_demo"
        }
        
    def load_model(self, model_name: str, model_type: str = "ncf"):
        """Charger un modèle TensorFlow"""
        try:
            model_key = f"{model_type}_{model_name}"
            
            if model_key in self.models:
                logger.info(f"Modèle {model_key} déjà chargé")
                return self.models[model_key]
            
            model_path = self.model_paths.get(model_key)
            
            if not model_path or not os.path.exists(model_path):
                logger.warning(f"Modèle {model_key} non trouvé à {model_path}")
                return None
            
            # Charger le modèle selon le type
            if model_type == "ncf":
                model = NCFModel(num_users=100, num_items=50)  # Valeurs par défaut
                model.load_model(model_path)
            else:
                logger.warning(f"Type de modèle {model_type} non supporté")
                return None
            
            self.models[model_key] = model
            logger.info(f"Modèle {model_key} chargé depuis {model_path}")
            
            return model
            
        except Exception as e:
            logger.error(f"Erreur chargement modèle {model_name}: {e}")
            return None
    
    def get_model(self, model_name: str, model_type: str = "ncf"):
        """Récupérer un modèle (charger si nécessaire)"""
        if model_name not in self.model_paths:
            return None
        
        model_key = f"{model_type}_{model_name}"
        
        if model_key not in self.models:
            return self.load_model(model_name, model_type)
        
        return self.models[model_key]

# Initialiser le gestionnaire
model_manager = TensorFlowModelManager()

def add_tensorflow_endpoints(app: FastAPI):
    """Ajouter les endpoints TensorFlow à l'API existante"""
    
    @app.post("/tf/recommend")
    async def get_tensorflow_recommendations(request: TensorFlowRecommendationRequest):
        """Obtenir des recommandations avec TensorFlow"""
        try:
            # Charger le modèle
            model = model_manager.get_model(request.category, request.model_type)
            
            if not model:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Modèle TensorFlow {request.model_type} non trouvé pour {request.category}"
                )
            
            # Générer les recommandations
            recommendations = model.recommend(
                request.user_id, 
                top_k=request.num_recommendations
            )
            
            # Formatter les résultats
            formatted_recs = [
                {
                    "item_id": item_id,
                    "score": float(score)
                }
                for item_id, score in recommendations
            ]
            
            result = {
                "user_id": request.user_id,
                "category": request.category,
                "model_type": request.model_type,
                "recommendations": formatted_recs,
                "num_recommendations": len(formatted_recs),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Recommandations TensorFlow générées pour {request.user_id}")
            return result
            
        except Exception as e:
            logger.error(f"Erreur recommandations TensorFlow: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/tf/predict")
    async def predict_tensorflow(request: TensorFlowPredictionRequest):
        """Prédire avec TensorFlow"""
        try:
            # Charger le modèle
            model = model_manager.get_model(request.category, request.model_type)
            
            if not model:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Modèle TensorFlow {request.model_type} non trouvé pour {request.category}"
                )
            
            # Prédire
            prediction = model.predict(request.user_id, request.item_id)
            
            result = {
                "user_id": request.user_id,
                "item_id": request.item_id,
                "category": request.category,
                "model_type": request.model_type,
                "prediction": float(prediction),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Prédiction TensorFlow générée pour {request.user_id} / {request.item_id}")
            return result
            
        except Exception as e:
            logger.error(f"Erreur prédiction TensorFlow: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/tf/models")
    async def list_tensorflow_models():
        """Lister les modèles TensorFlow disponibles"""
        try:
            models_info = []
            
            for model_key, model in model_manager.models.items():
                models_info.append({
                    "model_key": model_key,
                    "model_type": model_key.split('_')[0],
                    "category": model_key.split('_')[1],
                    "status": "loaded"
                })
            
            # Modèles disponibles mais non chargés
            for model_key, model_path in model_manager.model_paths.items():
                if model_key not in model_manager.models:
                    model_type, category = model_key.split('_')
                    if os.path.exists(model_path):
                        models_info.append({
                            "model_key": model_key,
                            "model_type": model_type,
                            "category": category,
                            "status": "available"
                        })
            
            return {
                "total_models": len(models_info),
                "models": models_info
            }
            
        except Exception as e:
            logger.error(f"Erreur liste modèles: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/tf/load_model")
    async def load_tensorflow_model(category: str, model_type: str = "ncf"):
        """Charger explicitement un modèle TensorFlow"""
        try:
            model = model_manager.load_model(category, model_type)
            
            if model:
                return {
                    "status": "success",
                    "model_key": f"{model_type}_{category}",
                    "message": f"Modèle {model_type} pour {category} chargé avec succès"
                }
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Impossible de charger le modèle {model_type} pour {category}"
                )
                
        except Exception as e:
            logger.error(f"Erreur chargement modèle: {e}")
            raise HTTPException(status_code=500, detail=str(e))

def main():
    """Fonction principale de démonstration"""
    print("=== DÉMONSTRATION INTÉGRATION API TENSORFLOW ===\n")
    
    # Simuler l'intégration
    print("Endpoints TensorFlow ajoutés:")
    print("  POST /tf/recommend - Recommandations NCF")
    print("  POST /tf/predict - Prédiction NCF")
    print("  GET /tf/models - Liste modèles")
    print("  POST /tf/load_model - Charger modèle")
    
    print("\nExemple d'utilisation:")
    print("  POST /tf/recommend")
    print('  {"user_id": "user_123", "category": "electronics", "model_type": "ncf"}')
    
    print("\nNote: Les modèles doivent être entraînés avant utilisation")
    print("Utilisez: python src/models/tensorflow_training_pipeline.py")

if __name__ == "__main__":
    main()
