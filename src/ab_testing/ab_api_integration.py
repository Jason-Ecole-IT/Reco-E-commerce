"""
Integration A/B Testing avec l'API de recommandation
Endpoints A/B testing pour l'API FastAPI
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
from datetime import datetime
import sys
import os

sys.path.append('.')

from src.ab_testing.ab_framework import ABTestingFramework, ABTestConfig, TestGroup

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialiser le framework A/B testing
ab_framework = ABTestingFramework()

# Modèles Pydantic
class ABTestCreate(BaseModel):
    test_name: str
    description: str
    traffic_split: float = 0.5
    min_sample_size: int = 1000
    metrics: List[str] = ["ctr", "conversion_rate"]

class ABUserAssignment(BaseModel):
    user_id: str
    test_id: str

class ABMetricTrack(BaseModel):
    user_id: str
    test_id: str
    metric_name: str
    value: float

class ABTestConfigUpdate(BaseModel):
    test_id: str
    status: Optional[str] = None
    end_date: Optional[datetime] = None

# Fonctions pour intégration avec l'API existante
def add_ab_testing_endpoints(app: FastAPI):
    """Ajouter les endpoints A/B testing à l'API existante"""
    
    @app.post("/ab/create_test")
    async def create_ab_test(request: ABTestCreate):
        """Créer un nouveau test A/B"""
        try:
            config = ABTestConfig(
                test_name=request.test_name,
                description=request.description,
                start_date=datetime.now(),
                traffic_split=request.traffic_split,
                min_sample_size=request.min_sample_size,
                metrics=request.metrics
            )
            
            test_id = ab_framework.create_test(config)
            
            return {
                "test_id": test_id,
                "test_name": request.test_name,
                "status": "created",
                "message": f"Test A/B '{request.test_name}' créé avec succès"
            }
            
        except Exception as e:
            logger.error(f"Erreur création test A/B: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/ab/assign_user")
    async def assign_user_to_group(request: ABUserAssignment):
        """Assigner un utilisateur à un groupe A ou B"""
        try:
            group = ab_framework.assign_user_to_group(request.user_id, request.test_id)
            
            return {
                "user_id": request.user_id,
                "test_id": request.test_id,
                "group": group.value,
                "assigned_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erreur assignment utilisateur: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/ab/track_metric")
    async def track_ab_metric(request: ABMetricTrack):
        """Tracker une métrique pour un utilisateur"""
        try:
            ab_framework.track_metric(
                request.user_id,
                request.test_id,
                request.metric_name,
                request.value
            )
            
            return {
                "status": "success",
                "message": f"Métrique {request.metric_name} trackée pour user {request.user_id}"
            }
            
        except Exception as e:
            logger.error(f"Erreur tracking métrique: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/ab/test/{test_id}")
    async def get_ab_test_summary(test_id: str):
        """Obtenir le résumé d'un test A/B"""
        try:
            summary = ab_framework.get_test_summary(test_id)
            return summary
            
        except Exception as e:
            logger.error(f"Erreur résumé test: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/ab/analyze/{test_id}")
    async def analyze_ab_test(test_id: str):
        """Analyser les résultats d'un test A/B"""
        try:
            results = ab_framework.analyze_test_results(test_id)
            return results
            
        except Exception as e:
            logger.error(f"Erreur analyse test: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/ab/stop/{test_id}")
    async def stop_ab_test(test_id: str):
        """Arrêter un test A/B"""
        try:
            results = ab_framework.stop_test(test_id)
            return results
            
        except Exception as e:
            logger.error(f"Erreur arrêt test: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/ab/list_tests")
    async def list_ab_tests():
        """Lister tous les tests A/B actifs"""
        try:
            tests = []
            for test_id, test_data in ab_framework.active_tests.items():
                summary = ab_framework.get_test_summary(test_id)
                tests.append(summary)
            
            return {
                "total_tests": len(tests),
                "tests": tests
            }
            
        except Exception as e:
            logger.error(f"Erreur liste tests: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/ab/sample_size")
    async def calculate_required_sample_size(effect_size: float, power: float = 0.8, alpha: float = 0.05):
        """Calculer la taille d'échantillon requise"""
        try:
            sample_size = ab_framework.calculate_sample_size(effect_size, power, alpha)
            
            return {
                "effect_size": effect_size,
                "power": power,
                "alpha": alpha,
                "required_sample_size": sample_size,
                "per_group": sample_size * 2  # A + B
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul sample size: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Fonction pour intégration avec les recommandations existantes
def get_recommendations_with_ab(user_id: str, category: str, test_id: str = None):
    """Obtenir des recommandations avec logique A/B testing"""
    try:
        if test_id and test_id in ab_framework.active_tests:
            # Assigner l'utilisateur à un groupe
            group = ab_framework.assign_user_to_group(user_id, test_id)
            
            if group == TestGroup.CONTROL:
                # Groupe A : Recommandations random (baseline)
                # Implémentation à ajouter
                return {"method": "random", "group": "A"}
            else:
                # Groupe B : Recommandations ML
                # Utiliser l'API existante
                return {"method": "ml", "group": "B"}
        else:
            # Pas de test A/B, utiliser ML par défaut
            return {"method": "ml", "group": "default"}
            
    except Exception as e:
        logger.error(f"Erreur recommandations A/B: {e}")
        return {"method": "ml", "group": "default"}

def main():
    """Fonction principale de démonstration"""
    # Simuler l'intégration
    print("=== DÉMONSTRATION INTÉGRATION API A/B TESTING ===\n")
    
    # Créer un test
    config = ABTestConfig(
        test_name="Demo Recommendation CTR Test",
        description="Test démo pour l'intégration API",
        start_date=datetime.now(),
        traffic_split=0.5,
        min_sample_size=50,
        metrics=["ctr", "conversion_rate"]
    )
    
    test_id = ab_framework.create_test(config)
    print(f"Test créé: {test_id}\n")
    
    # Simuler des appels API
    print("Simulation appels API:")
    
    # Assignment utilisateur
    assignment = {
        "user_id": "user_123",
        "test_id": test_id
    }
    print(f"POST /ab/assign_user: {assignment}")
    group = ab_framework.assign_user_to_group(assignment["user_id"], assignment["test_id"])
    print(f"→ Group: {group.value}\n")
    
    # Tracking métrique
    metric = {
        "user_id": "user_123",
        "test_id": test_id,
        "metric_name": "ctr",
        "value": 0.25
    }
    print(f"POST /ab/track_metric: {metric}")
    ab_framework.track_metric(metric["user_id"], metric["test_id"], metric["metric_name"], metric["value"])
    print("→ Metric tracked\n")
    
    # Analyse
    print(f"POST /ab/analyze/{test_id}:")
    results = ab_framework.analyze_test_results(test_id)
    print(f"→ Status: {results.get('overall_status', 'unknown')}\n")
    
    # Résumé
    print(f"GET /ab/test/{test_id}:")
    summary = ab_framework.get_test_summary(test_id)
    print(f"→ Total assignments: {summary['total_assignments']}")
    print(f"→ Group A: {summary['group_a_count']}")
    print(f"→ Group B: {summary['group_b_count']}")

if __name__ == "__main__":
    main()
