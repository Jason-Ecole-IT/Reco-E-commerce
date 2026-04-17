"""
Framework A/B Testing avec tests statistiques
Implementation complète pour tests de significance
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import json
import logging
from datetime import datetime
import hashlib
from dataclasses import dataclass
from enum import Enum

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestGroup(Enum):
    """Groupes de test A/B"""
    CONTROL = "A"  # Groupe contrôle (recommandations random)
    TREATMENT = "B"  # Groupe traitement (recommandations ML)

@dataclass
class ABTestConfig:
    """Configuration du test A/B"""
    test_name: str
    description: str
    start_date: datetime
    end_date: Optional[datetime] = None
    traffic_split: float = 0.5  # 50% A, 50% B
    min_sample_size: int = 1000
    significance_level: float = 0.05
    metrics: List[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["ctr", "conversion_rate", "avg_session_duration"]

@dataclass
class UserAssignment:
    """Assignment utilisateur"""
    user_id: str
    group: TestGroup
    assigned_at: datetime
    test_id: str

class ABTestingFramework:
    """Framework A/B Testing avec tests statistiques"""
    
    def __init__(self):
        """Initialiser le framework"""
        self.active_tests = {}
        self.user_assignments = {}
        self.metrics_data = {}
        self.test_results = {}
        
    def create_test(self, config: ABTestConfig) -> str:
        """Créer un nouveau test A/B"""
        try:
            test_id = self._generate_test_id(config.test_name)
            
            self.active_tests[test_id] = {
                "config": config,
                "status": "active",
                "created_at": datetime.now(),
                "assignments": {},
                "metrics": {metric: {"A": [], "B": []} for metric in config.metrics}
            }
            
            logger.info(f"Test A/B créé: {config.test_name} (ID: {test_id})")
            return test_id
            
        except Exception as e:
            logger.error(f"Erreur création test A/B: {e}")
            raise
    
    def assign_user_to_group(self, user_id: str, test_id: str) -> TestGroup:
        """Assigner un utilisateur à un groupe A ou B"""
        try:
            if test_id not in self.active_tests:
                raise ValueError(f"Test {test_id} non trouvé")
            
            # Vérifier si l'utilisateur est déjà assigné
            if user_id in self.active_tests[test_id]["assignments"]:
                return self.active_tests[test_id]["assignments"][user_id]
            
            # Assigner de manière déterministe (hash-based)
            config = self.active_tests[test_id]["config"]
            
            # Hash de l'user_id pour assignment consistent
            user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            hash_value = (user_hash % 100) / 100.0
            
            if hash_value < config.traffic_split:
                group = TestGroup.CONTROL
            else:
                group = TestGroup.TREATMENT
            
            # Enregistrer l'assignment
            self.active_tests[test_id]["assignments"][user_id] = group
            
            logger.debug(f"User {user_id} assigné au groupe {group.value}")
            return group
            
        except Exception as e:
            logger.error(f"Erreur assignment utilisateur: {e}")
            # Fallback: random assignment
            return TestGroup.CONTROL if np.random.random() < 0.5 else TestGroup.TREATMENT
    
    def track_metric(self, user_id: str, test_id: str, metric_name: str, value: float):
        """Tracker une métrique pour un utilisateur"""
        try:
            if test_id not in self.active_tests:
                raise ValueError(f"Test {test_id} non trouvé")
            
            # Récupérer le groupe de l'utilisateur
            group = self.assign_user_to_group(user_id, test_id)
            
            # Enregistrer la métrique
            if metric_name not in self.active_tests[test_id]["metrics"]:
                self.active_tests[test_id]["metrics"][metric_name] = {
                    "A": [],
                    "B": []
                }
            
            self.active_tests[test_id]["metrics"][metric_name][group.value].append({
                "user_id": user_id,
                "value": value,
                "timestamp": datetime.now()
            })
            
            logger.debug(f"Métrique {metric_name}={value} trackée pour user {user_id} (groupe {group.value})")
            
        except Exception as e:
            logger.error(f"Erreur tracking métrique: {e}")
    
    def calculate_sample_size(self, effect_size: float, power: float = 0.8, alpha: float = 0.05) -> int:
        """Calculer la taille d'échantillon nécessaire"""
        try:
            # Formule pour test de proportions
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = stats.norm.ppf(power)
            
            p1 = 0.1  # Baseline conversion rate (10%)
            p2 = p1 + effect_size
            
            n = (z_alpha + z_beta)**2 * (p1*(1-p1) + p2*(1-p2)) / (p1 - p2)**2
            
            return int(np.ceil(n))
            
        except Exception as e:
            logger.error(f"Erreur calcul sample size: {e}")
            return 1000
    
    def perform_t_test(self, group_a: List[float], group_b: List[float]) -> Dict:
        """Effectuer un t-test pour comparer deux groupes"""
        try:
            if len(group_a) < 30 or len(group_b) < 30:
                logger.warning("Sample size < 30, using non-parametric test")
                return self.perform_mann_whitney_test(group_a, group_b)
            
            # T-test à deux échantillons
            t_stat, p_value = stats.ttest_ind(group_a, group_b)
            
            # Moyennes et écarts-types
            mean_a = np.mean(group_a)
            mean_b = np.mean(group_b)
            std_a = np.std(group_a)
            std_b = np.std(group_b)
            
            # Confidence intervals (95%)
            ci_a = stats.t.interval(0.95, len(group_a)-1, loc=mean_a, scale=std_a/np.sqrt(len(group_a)))
            ci_b = stats.t.interval(0.95, len(group_b)-1, loc=mean_b, scale=std_b/np.sqrt(len(group_b)))
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(group_a)-1)*std_a**2 + (len(group_b)-1)*std_b**2) / (len(group_a)+len(group_b)-2))
            cohens_d = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0
            
            # Interprétation
            is_significant = p_value < 0.05
            improvement = ((mean_b - mean_a) / mean_a * 100) if mean_a > 0 else 0
            
            return {
                "test_type": "t_test",
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "is_significant": is_significant,
                "mean_a": float(mean_a),
                "mean_b": float(mean_b),
                "std_a": float(std_a),
                "std_b": float(std_b),
                "ci_a": [float(ci_a[0]), float(ci_a[1])],
                "ci_b": [float(ci_b[0]), float(ci_b[1])],
                "cohens_d": float(cohens_d),
                "improvement_percent": float(improvement),
                "sample_size_a": len(group_a),
                "sample_size_b": len(group_b)
            }
            
        except Exception as e:
            logger.error(f"Erreur t-test: {e}")
            return {"error": str(e)}
    
    def perform_mann_whitney_test(self, group_a: List[float], group_b: List[float]) -> Dict:
        """Effectuer un test de Mann-Whitney U (non-paramétrique)"""
        try:
            u_stat, p_value = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
            
            mean_a = np.mean(group_a)
            mean_b = np.mean(group_b)
            improvement = ((mean_b - mean_a) / mean_a * 100) if mean_a > 0 else 0
            
            return {
                "test_type": "mann_whitney_u",
                "u_statistic": float(u_stat),
                "p_value": float(p_value),
                "is_significant": p_value < 0.05,
                "mean_a": float(mean_a),
                "mean_b": float(mean_b),
                "improvement_percent": float(improvement),
                "sample_size_a": len(group_a),
                "sample_size_b": len(group_b)
            }
            
        except Exception as e:
            logger.error(f"Erreur Mann-Whitney test: {e}")
            return {"error": str(e)}
    
    def perform_chi_square_test(self, observed_a: Dict, observed_b: Dict) -> Dict:
        """Effectuer un test du chi-carré pour données catégorielles"""
        try:
            # Créer la table de contingence
            all_categories = set(list(observed_a.keys()) + list(observed_b.keys()))
            
            contingency_table = []
            for category in all_categories:
                contingency_table.append([
                    observed_a.get(category, 0),
                    observed_b.get(category, 0)
                ])
            
            # Chi-square test
            chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            return {
                "test_type": "chi_square",
                "chi2_statistic": float(chi2_stat),
                "p_value": float(p_value),
                "degrees_of_freedom": int(dof),
                "is_significant": p_value < 0.05,
                "contingency_table": contingency_table,
                "expected_frequencies": expected.tolist()
            }
            
        except Exception as e:
            logger.error(f"Erreur chi-square test: {e}")
            return {"error": str(e)}
    
    def analyze_test_results(self, test_id: str) -> Dict:
        """Analyser les résultats complets d'un test A/B"""
        try:
            if test_id not in self.active_tests:
                raise ValueError(f"Test {test_id} non trouvé")
            
            test_data = self.active_tests[test_id]
            config = test_data["config"]
            
            results = {
                "test_id": test_id,
                "test_name": config.test_name,
                "analysis_date": datetime.now().isoformat(),
                "metrics_analysis": {},
                "overall_status": "insufficient_data"
            }
            
            # Analyser chaque métrique
            for metric_name in config.metrics:
                if metric_name in test_data["metrics"]:
                    metric_data = test_data["metrics"][metric_name]
                    logger.info(f"Metric data type for {metric_name}: {type(metric_data)}")
                    
                    if metric_data is not None and isinstance(metric_data, dict):
                        group_a_values = [item["value"] for item in metric_data.get("A", [])]
                        group_b_values = [item["value"] for item in metric_data.get("B", [])]
                    else:
                        logger.error(f"Metric data for {metric_name} is not a dict: {metric_data}")
                        group_a_values = []
                        group_b_values = []
                    
                    if len(group_a_values) >= config.min_sample_size and len(group_b_values) >= config.min_sample_size:
                        # Effectuer le test statistique
                        test_result = self.perform_t_test(group_a_values, group_b_values)
                        test_result["metric_name"] = metric_name
                        test_result["sample_size_sufficient"] = True
                        
                        results["metrics_analysis"][metric_name] = test_result
                    else:
                        results["metrics_analysis"][metric_name] = {
                            "metric_name": metric_name,
                            "sample_size_sufficient": False,
                            "sample_size_a": len(group_a_values),
                            "sample_size_b": len(group_b_values),
                            "required_sample_size": config.min_sample_size,
                            "message": "Sample size insufficient for statistical significance"
                        }
            
            # Déterminer le statut global
            significant_metrics = [
                m for m in results["metrics_analysis"].values()
                if isinstance(m, dict) and m.get("is_significant", False) and m.get("sample_size_sufficient", False)
            ]
            
            if significant_metrics:
                results["overall_status"] = "significant_results"
                results["significant_metrics_count"] = len(significant_metrics)
            elif len(results["metrics_analysis"]) > 0:
                results["overall_status"] = "no_significant_results"
            
            # Sauvegarder les résultats
            self.test_results[test_id] = results
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur analyse résultats: {e}")
            return {"error": str(e)}
    
    def get_test_summary(self, test_id: str) -> Dict:
        """Obtenir un résumé du test"""
        try:
            if test_id not in self.active_tests:
                raise ValueError(f"Test {test_id} non trouvé")
            
            test_data = self.active_tests[test_id]
            config = test_data["config"]
            
            summary = {
                "test_id": test_id,
                "test_name": config.test_name,
                "description": config.description,
                "status": test_data["status"],
                "created_at": test_data["created_at"].isoformat(),
                "traffic_split": config.traffic_split,
                "total_assignments": len(test_data["assignments"]),
                "group_a_count": sum(1 for g in test_data["assignments"].values() if g == TestGroup.CONTROL),
                "group_b_count": sum(1 for g in test_data["assignments"].values() if g == TestGroup.TREATMENT),
                "metrics_tracked": list(test_data["metrics"].keys())
            }
            
            # Ajouter les résultats si disponibles
            if test_id in self.test_results:
                summary["results"] = self.test_results[test_id]
            
            return summary
            
        except Exception as e:
            logger.error(f"Erreur résumé test: {e}")
            return {"error": str(e)}
    
    def _generate_test_id(self, test_name: str) -> str:
        """Générer un ID de test unique"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_name = test_name.lower().replace(" ", "_").replace("-", "_")
        return f"ab_{clean_name}_{timestamp}"
    
    def stop_test(self, test_id: str):
        """Arrêter un test A/B"""
        try:
            if test_id not in self.active_tests:
                raise ValueError(f"Test {test_id} non trouvé")
            
            self.active_tests[test_id]["status"] = "stopped"
            self.active_tests[test_id]["stopped_at"] = datetime.now()
            
            # Analyser les résultats finaux
            results = self.analyze_test_results(test_id)
            
            logger.info(f"Test {test_id} arrêté et analysé")
            return results
            
        except Exception as e:
            logger.error(f"Erreur arrêt test: {e}")
            raise

def main():
    """Fonction principale de démonstration"""
    framework = ABTestingFramework()
    
    # Créer un test A/B
    config = ABTestConfig(
        test_name="Recommendation CTR Test",
        description="Test CTR des recommandations ML vs Random",
        start_date=datetime.now(),
        traffic_split=0.5,
        min_sample_size=100,
        metrics=["ctr", "conversion_rate"]
    )
    
    test_id = framework.create_test(config)
    
    print(f"Test créé: {test_id}")
    
    # Simuler des assignments
    for i in range(100):
        user_id = f"user_{i}"
        group = framework.assign_user_to_group(user_id, test_id)
        print(f"User {user_id} -> Group {group.value}")
    
    # Simuler des métriques
    np.random.seed(42)
    for i in range(100):
        user_id = f"user_{i}"
        
        # CTR: Groupe B (ML) meilleur que groupe A (random)
        group = framework.assign_user_to_group(user_id, test_id)
        if group == TestGroup.CONTROL:
            ctr = np.random.beta(2, 8)  # ~20% CTR
        else:
            ctr = np.random.beta(3, 7)  # ~30% CTR
        
        framework.track_metric(user_id, test_id, "ctr", ctr)
    
    # Analyser les résultats
    results = framework.analyze_test_results(test_id)
    
    print("\n=== RÉSULTATS TEST A/B ===")
    print(json.dumps(results, indent=2))
    
    # Résumé
    summary = framework.get_test_summary(test_id)
    print("\n=== RÉSUMÉ ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
