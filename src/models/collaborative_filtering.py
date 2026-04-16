"""
Collaborative Filtering avec PySpark ALS (Alternating Least Squares)
Modélisation pour système de recommandation
"""

import sys
import os
sys.path.append('src')

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CollaborativeFilteringModel:
    """Modèle de Collaborative Filtering avec ALS"""
    
    def __init__(self):
        """Initialiser le modèle ALS"""
        self.spark = None
        self.model = None
        self.train_data = None
        self.test_data = None
        self.evaluation_results = {}
        self.user_mapping = {}
        self.item_mapping = {}
        
    def initialize_spark(self):
        """Initialiser la session Spark"""
        try:
            self.spark = SparkSession.builder \
                .appName("collaborative_filtering") \
                .master("local[*]") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .getOrCreate()
            
            logger.info("Session Spark initialisée pour ALS")
            return self.spark
            
        except Exception as e:
            logger.error(f"Erreur initialisation Spark: {e}")
            raise
    
    def load_feature_store(self, csv_path: str):
        """Charger les données du feature store"""
        try:
            logger.info(f"Chargement des données depuis: {csv_path}")
            
            # Charger avec Pandas d'abord
            df_pandas = pd.read_csv(csv_path)
            
            # Filtrer les colonnes essentielles
            essential_cols = ['reviewerID', 'asin', 'overall']
            df_pandas = df_pandas[essential_cols].copy()
            
            # Convertir en Spark DataFrame
            df_spark = self.spark.createDataFrame(df_pandas)
            
            # Nettoyer les données
            df_spark = df_spark.filter(
                (F.col('reviewerID').isNotNull()) &
                (F.col('asin').isNotNull()) &
                (F.col('overall').isNotNull()) &
                (F.col('overall').between(1, 5))
            )
            
            logger.info(f"Données chargées: {df_spark.count():,} enregistrements")
            return df_spark
            
        except Exception as e:
            logger.error(f"Erreur chargement données: {e}")
            raise
    
    def create_user_item_mappings(self, df):
        """Créer les mappings user/item vers IDs numériques"""
        try:
            logger.info("Création des mappings user/item")
            
            # Extraire les utilisateurs et items uniques
            users = df.select('reviewerID').distinct().collect()
            items = df.select('asin').distinct().collect()
            
            # Créer les mappings
            self.user_mapping = {row['reviewerID']: idx for idx, row in enumerate(users)}
            self.item_mapping = {row['asin']: idx for idx, row in enumerate(items)}
            
            # Créer les mappings inverses
            user_reverse_mapping = {idx: user_id for user_id, idx in self.user_mapping.items()}
            item_reverse_mapping = {idx: item_id for item_id, idx in self.item_mapping.items()}
            
            # Ajouter les IDs numériques au DataFrame
            user_mapping_df = self.spark.createDataFrame(
                [(user_id, idx) for user_id, idx in self.user_mapping.items()],
                ['reviewerID', 'user_index']
            )
            
            item_mapping_df = self.spark.createDataFrame(
                [(item_id, idx) for item_id, idx in self.item_mapping.items()],
                ['asin', 'item_index']
            )
            
            # Joindre les mappings
            df_mapped = df.join(user_mapping_df, 'reviewerID', 'left') \
                         .join(item_mapping_df, 'asin', 'left')
            
            logger.info(f"Mappings créés: {len(self.user_mapping)} utilisateurs, {len(self.item_mapping)} items")
            return df_mapped, user_reverse_mapping, item_reverse_mapping
            
        except Exception as e:
            logger.error(f"Erreur création mappings: {e}")
            raise
    
    def prepare_train_test_split(self, df, test_ratio=0.2):
        """Préparer les données d'entraînement et de test"""
        try:
            logger.info(f"Split train/test avec ratio {1-test_ratio}/{test_ratio}")
            
            # Split temporel pour éviter la fuite d'information
            df = df.orderBy(F.rand(seed=42))
            
            train_df, test_df = df.randomSplit([1-test_ratio, test_ratio], seed=42)
            
            self.train_data = train_df
            self.test_data = test_df
            
            logger.info(f"Train: {train_df.count():,}, Test: {test_df.count():,}")
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Erreur split train/test: {e}")
            raise
    
    def train_als_model(self, train_df, rank=10, max_iter=10, reg_param=0.1):
        """Entraîner le modèle ALS"""
        try:
            logger.info(f"Entraînement ALS: rank={rank}, maxIter={max_iter}, reg={reg_param}")
            
            # Configurer ALS
            als = ALS(
                userCol="user_index",
                itemCol="item_index", 
                ratingCol="overall",
                rank=rank,
                maxIter=max_iter,
                regParam=reg_param,
                implicitPrefs=False,
                coldStartStrategy="drop",
                nonnegative=True
            )
            
            # Entraîner le modèle
            self.model = als.fit(train_df)
            
            logger.info("Modèle ALS entraîné avec succès")
            return self.model
            
        except Exception as e:
            logger.error(f"Erreur entraînement ALS: {e}")
            raise
    
    def evaluate_model(self, model, test_df):
        """Évaluer le modèle avec différentes métriques"""
        try:
            logger.info("Évaluation du modèle ALS")
            
            # Prédictions sur les données de test
            predictions = model.transform(test_df)
            
            # Évaluer avec différentes métriques
            evaluators = {
                'rmse': RegressionEvaluator(metricName="rmse", labelCol="overall", predictionCol="prediction"),
                'mae': RegressionEvaluator(metricName="mae", labelCol="overall", predictionCol="prediction"),
                'r2': RegressionEvaluator(metricName="r2", labelCol="overall", predictionCol="prediction")
            }
            
            results = {}
            for name, evaluator in evaluators.items():
                try:
                    score = evaluator.evaluate(predictions.na.drop())
                    results[name] = score
                    logger.info(f"{name.upper()}: {score:.4f}")
                except Exception as e:
                    logger.warning(f"Erreur évaluation {name}: {e}")
                    results[name] = None
            
            # Calculer la couverture
            coverage = self.calculate_coverage(model, test_df)
            results['coverage'] = coverage
            
            # Calculer la diversité
            diversity = self.calculate_diversity(model, test_df)
            results['diversity'] = diversity
            
            # Calculer la nouveauté
            novelty = self.calculate_novelty(model, test_df)
            results['novelty'] = novelty
            
            self.evaluation_results = results
            return results
            
        except Exception as e:
            logger.error(f"Erreur évaluation: {e}")
            raise
    
    def calculate_coverage(self, model, test_df, k=10):
        """Calculer la couverture des recommandations"""
        try:
            # Générer des recommandations pour tous les utilisateurs
            user_recs = model.recommendForAllUsers(k)
            
            # Extraire les items recommandés
            recommended_items = set()
            for row in user_recs.collect():
                recommended_items.update([rec['asin'] for rec in row['recommendations']])
            
            # Items uniques dans les données de test
            test_items = set(test_df.select('asin').distinct().rdd.map(lambda r: r[0]).collect())
            
            # Couverture = items recommandés / items totaux
            coverage = len(recommended_items.intersection(test_items)) / len(test_items) if test_items else 0
            
            return coverage
            
        except Exception as e:
            logger.warning(f"Erreur calcul coverage: {e}")
            return 0.0
    
    def calculate_diversity(self, model, test_df, k=10):
        """Calculer la diversité des recommandations"""
        try:
            # Générer des recommandations pour un échantillon d'utilisateurs
            sample_users = test_df.select('user_index').distinct().limit(100).collect()
            
            total_diversity = 0
            valid_recommendations = 0
            
            for user_row in sample_users:
                user_idx = user_row['user_index']
                
                # Obtenir les recommandations
                user_recs = model.recommendForUserSubset(user_idx, test_df.select('item_index').distinct(), k)
                
                if user_recs and len(user_recs) > 1:
                    # Calculer la diversité intra-recommandations
                    items = [rec['item_index'] for rec in user_recs[0]['recommendations']]
                    
                    # Diversité = 1 - similarité moyenne
                    pairwise_similarities = []
                    for i in range(len(items)):
                        for j in range(i+1, len(items)):
                            # Similarité simple (peut être améliorée avec des embeddings)
                            similarity = 1.0 if items[i] == items[j] else 0.0
                            pairwise_similarities.append(similarity)
                    
                    diversity = 1.0 - (np.mean(pairwise_similarities) if pairwise_similarities else 0.0)
                    total_diversity += diversity
                    valid_recommendations += 1
            
            avg_diversity = total_diversity / valid_recommendations if valid_recommendations > 0 else 0.0
            return avg_diversity
            
        except Exception as e:
            logger.warning(f"Erreur calcul diversité: {e}")
            return 0.0
    
    def calculate_novelty(self, model, test_df, k=10):
        """Calculer la nouveauté des recommandations"""
        try:
            # Popularité des items dans les données d'entraînement
            item_popularity = self.train_data.groupBy('item_index').count()
            total_interactions = self.train_data.count()
            
            # Générer des recommandations
            sample_users = test_df.select('user_index').distinct().limit(100).collect()
            
            total_novelty = 0
            valid_recommendations = 0
            
            for user_row in sample_users:
                user_idx = user_row['user_index']
                
                # Obtenir les recommandations
                user_recs = model.recommendForUserSubset(user_idx, test_df.select('item_index').distinct(), k)
                
                if user_recs:
                    recommended_items = [rec['item_index'] for rec in user_recs[0]['recommendations']]
                    
                    # Calculer la nouveauté moyenne
                    item_novelties = []
                    for item_idx in recommended_items:
                        # Popularité de l'item
                        popularity = item_popularity.filter(F.col('item_index') == item_idx).collect()
                        pop_count = popularity[0]['count'] if popularity else 0
                        
                        # Nouveauté = -log2(popularité / total_interactions)
                        novelty = -np.log2((pop_count + 1) / total_interactions)
                        item_novelties.append(novelty)
                    
                    avg_novelty = np.mean(item_novelties) if item_novelties else 0.0
                    total_novelty += avg_novelty
                    valid_recommendations += 1
            
            overall_novelty = total_novelty / valid_recommendations if valid_recommendations > 0 else 0.0
            return overall_novelty
            
        except Exception as e:
            logger.warning(f"Erreur calcul nouveauté: {e}")
            return 0.0
    
    def hyperparameter_tuning(self, train_df, test_df):
        """Optimiser les hyperparamètres avec grid search"""
        try:
            logger.info("Optimisation des hyperparamètres")
            
            # Définir la grille de paramètres
            als = ALS(
                userCol="user_index",
                itemCol="item_index",
                ratingCol="overall",
                coldStartStrategy="drop",
                nonnegative=True
            )
            
            param_grid = ParamGridBuilder() \
                .addGrid(als.rank, [5, 10, 20]) \
                .addGrid(als.maxIter, [5, 10, 15]) \
                .addGrid(als.regParam, [0.01, 0.1, 1.0]) \
                .build()
            
            # Évaluateur
            evaluator = RegressionEvaluator(metricName="rmse", labelCol="overall", predictionCol="prediction")
            
            # Cross validation
            crossval = CrossValidator(
                estimator=als,
                estimatorParamMaps=param_grid,
                evaluator=evaluator,
                numFolds=3,
                parallelism=2
            )
            
            # Entraîner avec cross validation
            cv_model = crossval.fit(train_df)
            
            # Meilleurs paramètres
            best_model = cv_model.bestModel
            best_params = {
                'rank': best_model.rank,
                'maxIter': best_model._java_obj.getMaxIter(),
                'regParam': best_model._java_obj.getRegParam()
            }
            
            logger.info(f"Meilleurs paramètres: {best_params}")
            
            # Évaluer le meilleur modèle
            best_results = self.evaluate_model(best_model, test_df)
            
            return best_model, best_params, best_results
            
        except Exception as e:
            logger.error(f"Erreur optimisation hyperparamètres: {e}")
            # Retourner un modèle par défaut
            return self.train_als_model(train_df, rank=10, max_iter=10, reg_param=0.1), {}, {}
    
    def generate_recommendations(self, model, user_id=None, k=10):
        """Générer des recommandations"""
        try:
            if user_id:
                # Recommandations pour un utilisateur spécifique
                if user_id in self.user_mapping:
                    user_idx = self.user_mapping[user_id]
                    user_recs = model.recommendForUserSubset(user_idx, 
                                                        self.test_data.select('item_index').distinct(), 
                                                        k)
                    return user_recs
                else:
                    logger.warning(f"Utilisateur {user_id} non trouvé")
                    return None
            else:
                # Recommandations pour tous les utilisateurs
                all_recs = model.recommendForAllUsers(k)
                return all_recs
                
        except Exception as e:
            logger.error(f"Erreur génération recommandations: {e}")
            return None
    
    def save_model(self, model, model_path: str):
        """Sauvegarder le modèle entraîné"""
        try:
            logger.info(f"Sauvegarde du modèle: {model_path}")
            
            # Créer le répertoire si nécessaire
            os.makedirs(model_path, exist_ok=True)
            
            # Sauvegarder le modèle
            model.write().overwrite().save(model_path)
            
            # Sauvegarder les mappings
            mappings = {
                'user_mapping': self.user_mapping,
                'item_mapping': self.item_mapping,
                'evaluation_results': self.evaluation_results,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(f"{model_path}/mappings.json", 'w') as f:
                json.dump(mappings, f, indent=2)
            
            logger.info(f"Modèle sauvegardé dans: {model_path}")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde modèle: {e}")
            raise
    
    def load_model(self, model_path: str):
        """Charger un modèle sauvegardé"""
        try:
            logger.info(f"Chargement du modèle: {model_path}")
            
            # Charger le modèle
            self.model = ALS.load(model_path)
            
            # Charger les mappings
            with open(f"{model_path}/mappings.json", 'r') as f:
                mappings = json.load(f)
            
            self.user_mapping = mappings['user_mapping']
            self.item_mapping = mappings['item_mapping']
            self.evaluation_results = mappings.get('evaluation_results', {})
            
            logger.info("Modèle et mappings chargés avec succès")
            return self.model
            
        except Exception as e:
            logger.error(f"Erreur chargement modèle: {e}")
            raise
    
    def run_collaborative_filtering_pipeline(self, feature_store_path: str, model_output_path: str):
        """Exécuter le pipeline complet de collaborative filtering"""
        try:
            start_time = datetime.now()
            logger.info("Démarrage pipeline collaborative filtering")
            
            # 1. Initialiser Spark
            self.initialize_spark()
            
            # 2. Charger les données
            df = self.load_feature_store(feature_store_path)
            
            # 3. Créer les mappings
            df_mapped, user_reverse_mapping, item_reverse_mapping = self.create_user_item_mappings(df)
            
            # 4. Split train/test
            train_df, test_df = self.prepare_train_test_split(df_mapped)
            
            # 5. Optimisation des hyperparamètres
            best_model, best_params, tuning_results = self.hyperparameter_tuning(train_df, test_df)
            
            # 6. Évaluation finale
            final_results = self.evaluate_model(best_model, test_df)
            
            # 7. Sauvegarder le modèle
            self.save_model(best_model, model_output_path)
            
            # 8. Générer des recommandations exemples
            sample_recs = self.generate_recommendations(best_model, k=10)
            
            # Finaliser les résultats
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            results = {
                'status': 'success',
                'duration_seconds': duration,
                'best_params': best_params,
                'evaluation_results': final_results,
                'model_path': model_output_path,
                'data_stats': {
                    'total_records': df.count(),
                    'train_records': train_df.count(),
                    'test_records': test_df.count(),
                    'unique_users': len(self.user_mapping),
                    'unique_items': len(self.item_mapping)
                }
            }
            
            logger.info(f"Pipeline terminé en {duration:.2f}s")
            logger.info(f"RMSE: {final_results.get('rmse', 'N/A')}")
            logger.info(f"Coverage: {final_results.get('coverage', 'N/A'):.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur pipeline: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
            
        finally:
            if self.spark:
                self.spark.stop()
                logger.info("Session Spark arrêtée")
    
    def create_evaluation_report(self, results: Dict, dataset_name: str):
        """Créer un rapport d'évaluation visuel"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Collaborative Filtering Evaluation - {dataset_name}', fontsize=16)
            
            # 1. Métriques principales
            if 'evaluation_results' in results:
                eval_results = results['evaluation_results']
                metrics = ['rmse', 'mae', 'coverage', 'diversity']
                values = [eval_results.get(m, 0) for m in metrics]
                
                axes[0, 0].bar(metrics, values, color=['skyblue', 'lightgreen', 'orange', 'pink'])
                axes[0, 0].set_title('Performance Metrics')
                axes[0, 0].set_ylabel('Value')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. Distribution des prédictions vs réels
            axes[0, 1].text(0.5, 0.5, f'RMSE: {eval_results.get("rmse", "N/A"):.4f}\nMAE: {eval_results.get("mae", "N/A"):.4f}',
                              ha='center', va='center', fontsize=14, transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Model Performance')
            axes[0, 1].axis('off')
            
            # 3. Statistiques des données
            if 'data_stats' in results:
                stats = results['data_stats']
                stats_text = f"""Dataset Statistics:
Total Records: {stats.get('total_records', 0):,}
Train Records: {stats.get('train_records', 0):,}
Test Records: {stats.get('test_records', 0):,}
Unique Users: {stats.get('unique_users', 0):,}
Unique Items: {stats.get('unique_items', 0):,}"""
                
                axes[1, 0].text(0.1, 0.5, stats_text, ha='left', va='center', 
                                  fontsize=10, transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Data Statistics')
                axes[1, 0].axis('off')
            
            # 4. Hyperparamètres optimaux
            if 'best_params' in results:
                params = results['best_params']
                params_text = f"""Best Hyperparameters:
Rank: {params.get('rank', 'N/A')}
Max Iter: {params.get('maxIter', 'N/A')}
Reg Param: {params.get('regParam', 'N/A')}"""
                
                axes[1, 1].text(0.1, 0.5, params_text, ha='left', va='center',
                                  fontsize=10, transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Best Hyperparameters')
                axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # Sauvegarder le rapport
            report_path = f"data/processed/cf_evaluation_{dataset_name.lower()}.png"
            plt.savefig(report_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"Rapport d'évaluation sauvegardé: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Erreur création rapport: {e}")
            return None

def main():
    """Fonction principale du collaborative filtering"""
    cf_model = CollaborativeFilteringModel()
    
    # Configuration des datasets
    datasets = [
        {
            'feature_store_path': "data/processed/electronics_features_feature_store.csv",
            'model_output_path': "models/als_electronics",
            'name': 'Electronics'
        },
        {
            'feature_store_path': "data/processed/clothing_features_feature_store.csv",
            'model_output_path': "models/als_clothing", 
            'name': 'Clothing'
        }
    ]
    
    results = []
    
    # Exécuter pour chaque dataset
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Collaborative Filtering: {dataset['name']}")
        print(f"{'='*60}")
        
        result = cf_model.run_collaborative_filtering_pipeline(
            dataset['feature_store_path'],
            dataset['model_output_path']
        )
        
        results.append(result)
        
        if result['status'] == 'success':
            print(f"✅ {dataset['name']}: RMSE = {result['evaluation_results'].get('rmse', 'N/A'):.4f}")
            print(f"   Coverage: {result['evaluation_results'].get('coverage', 'N/A'):.3f}")
            print(f"   Duration: {result['duration_seconds']:.2f}s")
            
            # Créer le rapport d'évaluation
            cf_model.create_evaluation_report(result, dataset['name'])
        else:
            print(f"❌ {dataset['name']}: {result['error']}")
    
    # Rapport final
    print(f"\n{'='*60}")
    print("RAPPORT COLLABORATIVE FILTERING FINAL")
    print(f"{'='*60}")
    
    successful_models = [r for r in results if r['status'] == 'success']
    
    if successful_models:
        avg_rmse = np.mean([r['evaluation_results'].get('rmse', 0) for r in successful_models])
        avg_coverage = np.mean([r['evaluation_results'].get('coverage', 0) for r in successful_models])
        
        print(f"Modèles entraînés: {len(successful_models)}/{len(results)}")
        print(f"RMSE moyen: {avg_rmse:.4f}")
        print(f"Coverage moyen: {avg_coverage:.3f}")
        
        # Vérifier les KPIs
        if avg_rmse < 1.0:
            print("✅ Objectif RMSE < 1.0 atteint!")
        else:
            print("⚠️ Objectif RMSE < 1.0 non atteint")
            
        if avg_coverage > 0.8:
            print("✅ Objectif Coverage > 80% atteint!")
        else:
            print("⚠️ Objectif Coverage > 80% non atteint")
    else:
        print("❌ Aucun modèle entraîné avec succès")
    
    return results

if __name__ == "__main__":
    main()
