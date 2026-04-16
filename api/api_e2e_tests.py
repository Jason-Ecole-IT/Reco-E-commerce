"""
Tests end-to-end pour l'API de recommandation
Tests complets de l'API avec validation des réponses
"""

import pytest
import requests
import json
import time
from typing import Dict, List
import pandas as pd
import numpy as np

# Configuration de l'API
API_BASE_URL = "http://localhost:8000"

class TestRecommendationAPI:
    """Tests end-to-end de l'API de recommandation"""
    
    def test_root_endpoint(self):
        """Test du endpoint racine"""
        response = requests.get(f"{API_BASE_URL}/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"
        
        print("✅ Root endpoint test passed")
    
    def test_health_check(self):
        """Test du health check"""
        response = requests.get(f"{API_BASE_URL}/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert data["status"] == "healthy"
        assert "models_loaded" in data
        assert "feature_stores_loaded" in data
        
        print("✅ Health check test passed")
    
    def test_recommendations_endpoint(self):
        """Test du endpoint de recommandations"""
        payload = {
            "user_id": "test_user_123",
            "category": "electronics",
            "num_recommendations": 5
        }
        
        response = requests.post(f"{API_BASE_URL}/recommend", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "user_id" in data
        assert data["user_id"] == payload["user_id"]
        assert "recommendations" in data
        assert len(data["recommendations"]) == payload["num_recommendations"]
        
        # Vérifier la structure des recommandations
        for rec in data["recommendations"]:
            assert "product_id" in rec
            assert "score" in rec
            assert isinstance(rec["score"], (int, float))
        
        print("✅ Recommendations endpoint test passed")
    
    def test_predict_rating_endpoint(self):
        """Test du endpoint de prédiction de note"""
        payload = {
            "user_id": "test_user_456",
            "product_id": "test_product_789",
            "category": "electronics"
        }
        
        response = requests.post(f"{API_BASE_URL}/predict_rating", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "user_id" in data
        assert "product_id" in data
        assert "predicted_rating" in data
        assert "confidence" in data
        
        # Vérifier que la note prédite est dans la plage valide
        assert 1 <= data["predicted_rating"] <= 5
        assert 0 <= data["confidence"] <= 1
        
        print("✅ Predict rating endpoint test passed")
    
    def test_batch_recommendations(self):
        """Test des recommandations batch"""
        payload = {
            "user_ids": ["user_1", "user_2", "user_3"],
            "category": "electronics",
            "num_recommendations": 3
        }
        
        response = requests.post(f"{API_BASE_URL}/batch_recommend", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert len(data["results"]) == len(payload["user_ids"])
        
        # Vérifier que chaque utilisateur a des résultats
        for result in data["results"]:
            assert "user_id" in result
            assert "status" in result
        
        print("✅ Batch recommendations test passed")
    
    def test_stats_endpoint(self):
        """Test du endpoint de statistiques"""
        response = requests.get(f"{API_BASE_URL}/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "feature_stores" in data
        assert "models_loaded" in data
        assert "redis_available" in data
        
        print("✅ Stats endpoint test passed")
    
    def test_cache_functionality(self):
        """Test de la fonctionnalité de cache"""
        payload = {
            "user_id": "cache_test_user",
            "category": "electronics",
            "num_recommendations": 5
        }
        
        # Première requête (cache miss)
        start_time = time.time()
        response1 = requests.post(f"{API_BASE_URL}/recommend", json=payload)
        time1 = time.time() - start_time
        
        assert response1.status_code == 200
        data1 = response1.json()
        
        # Deuxième requête (cache hit)
        start_time = time.time()
        response2 = requests.post(f"{API_BASE_URL}/recommend", json=payload)
        time2 = time.time() - start_time
        
        assert response2.status_code == 200
        data2 = response2.json()
        
        # Les résultats devraient être identiques
        assert data1["recommendations"] == data2["recommendations"]
        
        # La deuxième requête devrait être plus rapide
        assert time2 < time1
        
        print("✅ Cache functionality test passed")
    
    def test_error_handling(self):
        """Test de la gestion des erreurs"""
        # Test avec catégorie invalide
        payload = {
            "user_id": "test_user",
            "category": "invalid_category",
            "num_recommendations": 5
        }
        
        response = requests.post(f"{API_BASE_URL}/recommend", json=payload)
        
        # Devrait retourner une erreur 404
        assert response.status_code == 404
        
        print("✅ Error handling test passed")
    
    def test_performance_benchmark(self):
        """Test de performance de l'API"""
        payload = {
            "user_id": "perf_test_user",
            "category": "electronics",
            "num_recommendations": 10
        }
        
        # Effectuer 10 requêtes
        times = []
        for _ in range(10):
            start_time = time.time()
            response = requests.post(f"{API_BASE_URL}/recommend", json=payload)
            times.append(time.time() - start_time)
            
            assert response.status_code == 200
        
        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)
        p99_time = np.percentile(times, 99)
        
        print(f"Performance Benchmark:")
        print(f"  Average: {avg_time*1000:.2f}ms")
        print(f"  P95: {p95_time*1000:.2f}ms")
        print(f"  P99: {p99_time*1000:.2f}ms")
        
        # Vérifier que le temps moyen est < 50ms (objectif)
        assert avg_time < 0.05, f"Average time {avg_time*1000:.2f}ms exceeds 50ms threshold"
        
        print("✅ Performance benchmark test passed")
    
    def test_data_consistency(self):
        """Test de la consistance des données"""
        payload = {
            "user_id": "consistency_test_user",
            "category": "electronics",
            "num_recommendations": 5
        }
        
        # Effectuer plusieurs requêtes pour le même utilisateur
        recommendations_sets = []
        for _ in range(3):
            response = requests.post(f"{API_BASE_URL}/recommend", json=payload)
            assert response.status_code == 200
            data = response.json()
            recommendations_sets.append(data["recommendations"])
        
        # Les recommandations devraient être cohérentes
        # (au moins les produits devraient être les mêmes)
        assert recommendations_sets[0] == recommendations_sets[1]
        assert recommendations_sets[1] == recommendations_sets[2]
        
        print("✅ Data consistency test passed")

def run_all_tests():
    """Exécuter tous les tests"""
    test_suite = TestRecommendationAPI()
    
    tests = [
        ("Root Endpoint", test_suite.test_root_endpoint),
        ("Health Check", test_suite.test_health_check),
        ("Recommendations", test_suite.test_recommendations_endpoint),
        ("Predict Rating", test_suite.test_predict_rating_endpoint),
        ("Batch Recommendations", test_suite.test_batch_recommendations),
        ("Stats", test_suite.test_stats_endpoint),
        ("Cache Functionality", test_suite.test_cache_functionality),
        ("Error Handling", test_suite.test_error_handling),
        ("Performance Benchmark", test_suite.test_performance_benchmark),
        ("Data Consistency", test_suite.test_data_consistency)
    ]
    
    print("="*60)
    print("DÉBUT DES TESTS END-TO-END API")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            failed += 1
    
    print("="*60)
    print(f"RÉSULTAT: {passed}/{len(tests)} tests passés")
    if failed > 0:
        print(f"❌ {failed} tests échoués")
    else:
        print("✅ Tous les tests passés avec succès!")
    print("="*60)
    
    return passed, failed

if __name__ == "__main__":
    run_all_tests()
