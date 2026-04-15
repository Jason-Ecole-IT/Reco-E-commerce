"""
Collaborative Filtering avec scikit-learn (compatible Windows)
Alternative à PySpark ALS pour environnement de développement
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CollaborativeFilteringSklearn:
    """Modèle de Collaborative Filtering avec scikit-learn"""
    
    def __init__(self):
        """Initialiser le modèle"""
        self.user_mapping = {}
        self.item_mapping = {}
        self.user_reverse_mapping = {}
        self.item_reverse_mapping = {}
        self.rating_matrix = None
        self.model = None
        self.evaluation_results = {}
        
    def load_data(self, csv_path: str):
        """Charger les données du feature store"""
        try:
            logger.info(f"Chargement des données depuis: {csv_path}")
            
            # Charger les données
            df = pd.read_csv(csv_path)
            
            # Filtrer les colonnes essentielles
            essential_cols = ['reviewerID', 'asin', 'overall']
            df = df[essential_cols].copy()
            
            # Nettoyer les données
            df = df.dropna(subset=essential_cols)
            df = df[df['overall'].between(1, 5)]
            
            logger.info(f"Données chargées: {len(df):,} enregistrements")
            return df
            
        except Exception as e:
            logger.error(f"Erreur chargement données: {e}")
            raise
    
    def create_user_item_mappings(self, df):
        """Créer les mappings user/item vers IDs numériques"""
        try:
            logger.info("Création des mappings user/item")
            
            # Extraire les utilisateurs et items uniques
            unique_users = df['reviewerID'].unique()
            unique_items = df['asin'].unique()
            
            # Créer les mappings
            self.user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
            self.item_mapping = {item_id: idx for idx, item_id in enumerate(unique_items)}
            
            # Créer les mappings inverses
            self.user_reverse_mapping = {idx: user_id for user_id, idx in self.user_mapping.items()}
            self.item_reverse_mapping = {idx: item_id for item_id, idx in self.item_mapping.items()}
            
            # Ajouter les IDs numériques au DataFrame
            df['user_index'] = df['reviewerID'].map(self.user_mapping)
            df['item_index'] = df['asin'].map(self.item_mapping)
            
            logger.info(f"Mappings créés: {len(self.user_mapping)} utilisateurs, {len(self.item_mapping)} items")
            return df
            
        except Exception as e:
            logger.error(f"Erreur création mappings: {e}")
            raise
    
    def create_rating_matrix(self, df):
        """Créer la matrice utilisateur-item"""
        try:
            logger.info("Création de la matrice utilisateur-item")
            
            # Créer la matrice sparse
            n_users = len(self.user_mapping)
            n_items = len(self.item_mapping)
            
            # Initialiser la matrice avec 0 (valeurs manquantes)
            rating_matrix = np.zeros((n_users, n_items))
            
            # Remplir la matrice
            for _, row in df.iterrows():
                user_idx = int(row['user_index'])
                item_idx = int(row['item_index'])
                rating = float(row['overall'])
                
                rating_matrix[user_idx, item_idx] = rating
            
            self.rating_matrix = rating_matrix
            
            logger.info(f"Matrice créée: {rating_matrix.shape}")
            return rating_matrix
            
        except Exception as e:
            logger.error(f"Erreur création matrice: {e}")
            raise
    
    def train_svd_model(self, train_matrix, n_components=50):
        """Entraîner le modèle SVD (Matrix Factorization)"""
        try:
            logger.info(f"Entraînement SVD avec {n_components} composantes")
            
            # Créer le modèle SVD
            self.model = TruncatedSVD(n_components=n_components, random_state=42)
            
            # Entraîner sur la matrice
            user_factors = self.model.fit_transform(train_matrix)
            item_factors = self.model.components_.T
            
            # Reconstruire la matrice
            predicted_matrix = np.dot(user_factors, item_factors.T)
            
            # Normaliser les prédictions entre 1 et 5
            predicted_matrix = np.clip(predicted_matrix, 1, 5)
            
            logger.info("Modèle SVD entraîné avec succès")
            return predicted_matrix, user_factors, item_factors
            
        except Exception as e:
            logger.error(f"Erreur entraînement SVD: {e}")
            raise
    
    def train_knn_model(self, train_matrix, n_neighbors=20):
        """Entraîner le modèle KNN pour neighborhood-based CF"""
        try:
            logger.info(f"Entraînement KNN avec {n_neighbors} voisins")
            
            # Créer le modèle KNN
            self.model = NearestNeighbors(
                n_neighbors=n_neighbors,
                metric='cosine',
                algorithm='brute'
            )
            
            # Entraîner sur les utilisateurs
            self.model.fit(train_matrix)
            
            logger.info("Modèle KNN entraîné avec succès")
            return self.model
            
        except Exception as e:
            logger.error(f"Erreur entraînement KNN: {e}")
            raise
    
    def prepare_train_test_split(self, df, test_ratio=0.2):
        """Préparer les données d'entraînement et de test"""
        try:
            logger.info(f"Split train/test avec ratio {1-test_ratio}/{test_ratio}")
            
            # Split aléatoire
            train_df, test_df = train_test_split(
                df, 
                test_size=test_ratio, 
                random_state=42
            )
            
            logger.info(f"Train: {len(train_df):,}, Test: {len(test_df):,}")
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Erreur split train/test: {e}")
            raise
    
    def evaluate_model(self, predicted_matrix, test_df):
        """Évaluer le modèle avec différentes métriques"""
        try:
            logger.info("Évaluation du modèle")
            
            predictions = []
            actuals = []
            
            # Prédictions pour les données de test
            for _, row in test_df.iterrows():
                user_idx = int(row['user_index'])
                item_idx = int(row['item_index'])
                actual_rating = float(row['overall'])
                
                # Prédiction
                if user_idx < predicted_matrix.shape[0] and item_idx < predicted_matrix.shape[1]:
                    predicted_rating = predicted_matrix[user_idx, item_idx]
                else:
                    predicted_rating = 3.0  # Valeur par défaut
                
                predictions.append(predicted_rating)
                actuals.append(actual_rating)
            
            # Calculer les métriques
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            
            # Calculer R²
            ss_res = np.sum((np.array(actuals) - np.array(predictions)) ** 2)
            ss_tot = np.sum((np.array(actuals) - np.mean(actuals)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Calculer la couverture
            coverage = self.calculate_coverage(predicted_matrix, test_df)
            
            # Calculer la diversité
            diversity = self.calculate_diversity(predicted_matrix, test_df)
            
            results = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'coverage': coverage,
                'diversity': diversity,
                'n_predictions': len(predictions)
            }
            
            self.evaluation_results = results
            
            logger.info(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            logger.info(f"Coverage: {coverage:.3f}, Diversity: {diversity:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur évaluation: {e}")
            raise
    
    def calculate_coverage(self, predicted_matrix, test_df, k=10):
        """Calculer la couverture des recommandations"""
        try:
            # Items uniques dans les données de test
            test_items = set(test_df['item_index'].unique())
            
            # Pour chaque utilisateur, obtenir les top-k items
            recommended_items = set()
            for user_idx in range(predicted_matrix.shape[0]):
                user_predictions = predicted_matrix[user_idx]
                top_k_indices = np.argsort(user_predictions)[-k:][::-1]
                recommended_items.update(top_k_indices)
            
            # Couverture = items recommandés ∩ items test / items test
            coverage = len(recommended_items.intersection(test_items)) / len(test_items) if test_items else 0
            
            return coverage
            
        except Exception as e:
            logger.warning(f"Erreur calcul coverage: {e}")
            return 0.0
    
    def calculate_diversity(self, predicted_matrix, test_df, k=10):
        """Calculer la diversité des recommandations"""
        try:
            # Échantillon d'utilisateurs
            sample_users = np.random.choice(
                predicted_matrix.shape[0], 
                size=min(100, predicted_matrix.shape[0]), 
                replace=False
            )
            
            total_diversity = 0
            valid_recommendations = 0
            
            for user_idx in sample_users:
                user_predictions = predicted_matrix[user_idx]
                top_k_indices = np.argsort(user_predictions)[-k:][::-1]
                
                if len(top_k_indices) > 1:
                    # Calculer la diversité intra-recommandations
                    # Diversité = 1 - similarité moyenne
                    diversity_score = 1.0 - (1.0 / len(top_k_indices))  # Simplifié
                    total_diversity += diversity_score
                    valid_recommendations += 1
            
            avg_diversity = total_diversity / valid_recommendations if valid_recommendations > 0 else 0.0
            return avg_diversity
            
        except Exception as e:
            logger.warning(f"Erreur calcul diversité: {e}")
            return 0.0
    
    def generate_recommendations(self, predicted_matrix, user_id=None, k=10):
        """Générer des recommandations"""
        try:
            if user_id:
                # Recommandations pour un utilisateur spécifique
                if user_id in self.user_mapping:
                    user_idx = self.user_mapping[user_id]
                    
                    if user_idx < predicted_matrix.shape[0]:
                        user_predictions = predicted_matrix[user_idx]
                        top_k_indices = np.argsort(user_predictions)[-k:][::-1]
                        
                        recommendations = []
                        for item_idx in top_k_indices:
                            if item_idx in self.item_reverse_mapping:
                                recommendations.append({
                                    'asin': self.item_reverse_mapping[item_idx],
                                    'predicted_rating': user_predictions[item_idx]
                                })
                        
                        return recommendations
                    else:
                        logger.warning(f"Utilisateur {user_id} hors limites")
                        return []
                else:
                    logger.warning(f"Utilisateur {user_id} non trouvé")
                    return []
            else:
                # Recommandations pour tous les utilisateurs
                all_recommendations = {}
                
                for user_idx in range(predicted_matrix.shape[0]):
                    user_predictions = predicted_matrix[user_idx]
                    top_k_indices = np.argsort(user_predictions)[-k:][::-1]
                    
                    recommendations = []
                    for item_idx in top_k_indices:
                        if item_idx in self.item_reverse_mapping:
                            recommendations.append({
                                'asin': self.item_reverse_mapping[item_idx],
                                'predicted_rating': user_predictions[item_idx]
                            })
                    
                    user_id = self.user_reverse_mapping.get(user_idx, f"user_{user_idx}")
                    all_recommendations[user_id] = recommendations
                
                return all_recommendations
                
        except Exception as e:
            logger.error(f"Erreur génération recommandations: {e}")
            return None
    
    def hyperparameter_tuning(self, train_df, test_df):
        """Optimiser les hyperparamètres"""
        try:
            logger.info("Optimisation des hyperparamètres")
            
            # Créer la matrice d'entraînement
            train_matrix = np.zeros((len(self.user_mapping), len(self.item_mapping)))
            for _, row in train_df.iterrows():
                user_idx = int(row['user_index'])
                item_idx = int(row['item_index'])
                rating = float(row['overall'])
                train_matrix[user_idx, item_idx] = rating
            
            # Grille de paramètres
            param_grid = [
                {'n_components': 10, 'name': 'SVD-10'},
                {'n_components': 25, 'name': 'SVD-25'},
                {'n_components': 50, 'name': 'SVD-50'},
                {'n_components': 100, 'name': 'SVD-100'}
            ]
            
            best_params = None
            best_rmse = float('inf')
            best_results = None
            
            for params in param_grid:
                logger.info(f"Test: {params['name']}")
                
                # Entraîner le modèle
                predicted_matrix, _, _ = self.train_svd_model(train_matrix, params['n_components'])
                
                # Évaluer
                results = self.evaluate_model(predicted_matrix, test_df)
                
                if results['rmse'] < best_rmse:
                    best_rmse = results['rmse']
                    best_params = params
                    best_results = results
            
            logger.info(f"Meilleurs paramètres: {best_params['name']}, RMSE: {best_rmse:.4f}")
            
            return best_params, best_results
            
        except Exception as e:
            logger.error(f"Erreur optimisation hyperparamètres: {e}")
            return {}, {}
    
    def save_model(self, model_path: str):
        """Sauvegarder le modèle"""
        try:
            logger.info(f"Sauvegarde du modèle: {model_path}")
            
            # Créer le répertoire
            import os
            os.makedirs(model_path, exist_ok=True)
            
            # Sauvegarder les mappings
            mappings = {
                'user_mapping': self.user_mapping,
                'item_mapping': self.item_mapping,
                'user_reverse_mapping': self.user_reverse_mapping,
                'item_reverse_mapping': self.item_reverse_mapping,
                'evaluation_results': self.evaluation_results,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(f"{model_path}/mappings.json", 'w') as f:
                json.dump(mappings, f, indent=2)
            
            # Sauvegarder le modèle SVD
            if self.model is not None:
                import joblib
                joblib.dump(self.model, f"{model_path}/svd_model.pkl")
            
            # Sauvegarder la matrice de prédictions
            if hasattr(self, 'predicted_matrix'):
                np.save(f"{model_path}/predicted_matrix.npy", self.predicted_matrix)
            
            logger.info(f"Modèle sauvegardé dans: {model_path}")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde modèle: {e}")
            raise
    
    def load_model(self, model_path: str):
        """Charger un modèle sauvegardé"""
        try:
            logger.info(f"Chargement du modèle: {model_path}")
            
            # Charger les mappings
            with open(f"{model_path}/mappings.json", 'r') as f:
                mappings = json.load(f)
            
            self.user_mapping = mappings['user_mapping']
            self.item_mapping = mappings['item_mapping']
            self.user_reverse_mapping = mappings['user_reverse_mapping']
            self.item_reverse_mapping = mappings['item_reverse_mapping']
            self.evaluation_results = mappings.get('evaluation_results', {})
            
            # Charger le modèle SVD
            import joblib
            import os
            if os.path.exists(f"{model_path}/svd_model.pkl"):
                self.model = joblib.load(f"{model_path}/svd_model.pkl")
            
            # Charger la matrice de prédictions
            if os.path.exists(f"{model_path}/predicted_matrix.npy"):
                self.predicted_matrix = np.load(f"{model_path}/predicted_matrix.npy")
            
            logger.info("Modèle chargé avec succès")
            return self.model
            
        except Exception as e:
            logger.error(f"Erreur chargement modèle: {e}")
            raise
    
    def run_collaborative_filtering_pipeline(self, feature_store_path: str, model_output_path: str):
        """Exécuter le pipeline complet"""
        try:
            start_time = datetime.now()
            logger.info("Démarrage pipeline collaborative filtering")
            
            # 1. Charger les données
            df = self.load_data(feature_store_path)
            
            # 2. Créer les mappings
            df = self.create_user_item_mappings(df)
            
            # 3. Split train/test
            train_df, test_df = self.prepare_train_test_split(df)
            
            # 4. Créer la matrice utilisateur-item
            train_matrix = np.zeros((len(self.user_mapping), len(self.item_mapping)))
            for _, row in train_df.iterrows():
                user_idx = int(row['user_index'])
                item_idx = int(row['item_index'])
                rating = float(row['overall'])
                train_matrix[user_idx, item_idx] = rating
            
            # 5. Optimisation des hyperparamètres
            best_params, best_results = self.hyperparameter_tuning(train_df, test_df)
            
            # 6. Entraîner le modèle final
            self.predicted_matrix, _, _ = self.train_svd_model(
                train_matrix, 
                best_params.get('n_components', 50)
            )
            
            # 7. Évaluation finale
            final_results = self.evaluate_model(self.predicted_matrix, test_df)
            
            # 8. Sauvegarder le modèle
            self.save_model(model_output_path)
            
            # 9. Générer des recommandations exemples
            sample_recs = self.generate_recommendations(
                self.predicted_matrix, 
                user_id=list(self.user_mapping.keys())[0], 
                k=10
            )
            
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
                    'total_records': len(df),
                    'train_records': len(train_df),
                    'test_records': len(test_df),
                    'unique_users': len(self.user_mapping),
                    'unique_items': len(self.item_mapping)
                },
                'sample_recommendations': sample_recs[:5] if sample_recs else []
            }
            
            logger.info(f"Pipeline terminé en {duration:.2f}s")
            logger.info(f"RMSE: {final_results.get('rmse', 'N/A'):.4f}")
            logger.info(f"Coverage: {final_results.get('coverage', 'N/A'):.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur pipeline: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def create_evaluation_report(self, results: Dict, dataset_name: str):
        """Créer un rapport d'évaluation visuel"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Collaborative Filtering Evaluation - {dataset_name}', fontsize=16)
            
            # 1. Métriques principales
            if 'evaluation_results' in results:
                eval_results = results['evaluation_results']
                metrics = ['rmse', 'mae', 'r2', 'coverage']
                values = [eval_results.get(m, 0) for m in metrics]
                
                axes[0, 0].bar(metrics, values, color=['skyblue', 'lightgreen', 'orange', 'pink'])
                axes[0, 0].set_title('Performance Metrics')
                axes[0, 0].set_ylabel('Value')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. Distribution des prédictions
            if hasattr(self, 'predicted_matrix'):
                predictions_flat = self.predicted_matrix.flatten()
                predictions_flat = predictions_flat[predictions_flat > 0]
                
                axes[0, 1].hist(predictions_flat, bins=50, alpha=0.7, color='lightblue')
                axes[0, 1].set_title('Predicted Ratings Distribution')
                axes[0, 1].set_xlabel('Predicted Rating')
                axes[0, 1].set_ylabel('Frequency')
            
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
            
            # 4. Recommandations exemples
            if 'sample_recommendations' in results:
                recs = results['sample_recommendations']
                if recs:
                    rec_text = "Sample Recommendations:\n"
                    for i, rec in enumerate(recs[:5]):
                        rec_text += f"{i+1}. {rec['asin']}: {rec['predicted_rating']:.2f}\n"
                    
                    axes[1, 1].text(0.1, 0.5, rec_text, ha='left', va='top',
                                      fontsize=9, transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title('Sample Recommendations')
                    axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # Sauvegarder le rapport
            report_path = f"data/processed/cf_evaluation_{dataset_name.lower()}_sklearn.png"
            plt.savefig(report_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"Rapport d'évaluation sauvegardé: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Erreur création rapport: {e}")
            return None

def main():
    """Fonction principale"""
    cf_model = CollaborativeFilteringSklearn()
    
    # Configuration des datasets
    datasets = [
        {
            'feature_store_path': "data/processed/electronics_features_feature_store.csv",
            'model_output_path': "models/svd_electronics",
            'name': 'Electronics'
        },
        {
            'feature_store_path': "data/processed/clothing_features_feature_store.csv",
            'model_output_path': "models/svd_clothing",
            'name': 'Clothing'
        }
    ]
    
    results = []
    
    # Exécuter pour chaque dataset
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Collaborative Filtering (SVD): {dataset['name']}")
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
