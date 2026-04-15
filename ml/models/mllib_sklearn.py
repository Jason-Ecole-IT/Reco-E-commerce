"""
MLlib avec scikit-learn (compatible Windows)
Pipelines ML avec hyperparameter tuning et cross-validation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedMLSklearn:
    """Pipeline ML avancé avec scikit-learn"""
    
    def __init__(self):
        """Initialiser le pipeline ML"""
        self.pipelines = {}
        self.models = {}
        self.evaluation_results = {}
        self.best_model = None
        self.best_pipeline = None
        self.label_encoders = {}
        
    def load_feature_store(self, csv_path: str):
        """Charger les données du feature store"""
        try:
            logger.info(f"Chargement des données depuis: {csv_path}")
            
            # Charger les données
            df = pd.read_csv(csv_path)
            
            # Sélectionner les features pertinentes
            feature_cols = [
                'overall', 'word_count', 'sentiment_score', 'helpfulness_ratio',
                'user_review_count', 'user_avg_rating', 'product_review_count', 
                'product_avg_rating', 'user_experience', 'product_popularity'
            ]
            
            # Vérifier les colonnes disponibles
            available_cols = [col for col in feature_cols if col in df.columns]
            df = df[available_cols].copy()
            
            # Nettoyer les données
            df = df.dropna()
            
            logger.info(f"Données chargées: {len(df):,} enregistrements")
            logger.info(f"Features disponibles: {available_cols}")
            
            return df, available_cols
            
        except Exception as e:
            logger.error(f"Erreur chargement données: {e}")
            raise
    
    def create_classification_target(self, df):
        """Créer la cible de classification"""
        try:
            logger.info("Création de la cible de classification")
            
            # Créer la cible binaire
            df['rating_category'] = (df['overall'] >= 4).astype(int)
            
            # Statistiques
            positive_count = df['rating_category'].sum()
            negative_count = len(df) - positive_count
            
            logger.info(f"Ratings positifs: {positive_count} ({positive_count/len(df):.1%})")
            logger.info(f"Ratings négatifs: {negative_count} ({negative_count/len(df):.1%})")
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur création cible: {e}")
            raise
    
    def prepare_features(self, df, feature_cols):
        """Préparer les features pour ML"""
        try:
            logger.info("Préparation des features")
            
            # Séparer les colonnes catégorielles et numériques
            categorical_cols = ['user_experience', 'product_popularity']
            categorical_cols = [col for col in categorical_cols if col in feature_cols]
            
            numeric_cols = [col for col in feature_cols if col not in categorical_cols and col != 'overall']
            
            # Encoder les colonnes catégorielles
            df_encoded = df.copy()
            for col in categorical_cols:
                if col in df_encoded.columns:
                    le = LabelEncoder()
                    df_encoded[f"{col}_encoded"] = le.fit_transform(df_encoded[col].astype(str))
                    self.label_encoders[col] = le
            
            # Sélectionner les features finales
            encoded_cols = [f"{col}_encoded" for col in categorical_cols]
            feature_cols_final = numeric_cols + encoded_cols
            
            # Supprimer les colonnes catégorielles originales
            df_features = df_encoded[feature_cols_final + ['overall', 'rating_category']].copy()
            
            # Nettoyer les valeurs infinies et NaN
            df_features = df_features.replace([np.inf, -np.inf], np.nan)
            df_features = df_features.dropna()
            
            logger.info(f"Features finales: {len(feature_cols_final)}")
            return df_features, feature_cols_final
            
        except Exception as e:
            logger.error(f"Erreur préparation features: {e}")
            raise
    
    def create_classification_models(self):
        """Créer les modèles de classification"""
        try:
            logger.info("Création des modèles de classification")
            
            models = {
                'logistic_regression': LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    n_jobs=-1
                ),
                'random_forest': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=8,
                    learning_rate=0.1,
                    random_state=42
                )
            }
            
            logger.info(f"Modèles de classification créés: {list(models.keys())}")
            return models
            
        except Exception as e:
            logger.error(f"Erreur création modèles classification: {e}")
            raise
    
    def create_regression_models(self):
        """Créer les modèles de régression"""
        try:
            logger.info("Création des modèles de régression")
            
            models = {
                'linear_regression': LinearRegression(
                    n_jobs=-1
                ),
                'random_forest_reg': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                ),
                'gradient_boosting_reg': GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=8,
                    learning_rate=0.1,
                    random_state=42
                )
            }
            
            logger.info(f"Modèles de régression créés: {list(models.keys())}")
            return models
            
        except Exception as e:
            logger.error(f"Erreur création modèles régression: {e}")
            raise
    
    def create_hyperparameter_grid(self, model_type='classification'):
        """Créer la grille d'hyperparamètres"""
        try:
            logger.info(f"Création grille d'hyperparamètres: {model_type}")
            
            if model_type == 'classification':
                param_grids = {
                    'random_forest': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [5, 10, 15],
                        'min_samples_leaf': [1, 5, 10]
                    },
                    'gradient_boosting': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [5, 8, 12],
                        'learning_rate': [0.05, 0.1, 0.2]
                    },
                    'logistic_regression': {
                        'C': [0.1, 1.0, 10.0],
                        'penalty': ['l1', 'l2']
                    }
                }
                
            elif model_type == 'regression':
                param_grids = {
                    'random_forest_reg': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [5, 10, 15],
                        'min_samples_leaf': [1, 5, 10]
                    },
                    'gradient_boosting_reg': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [5, 8, 12],
                        'learning_rate': [0.05, 0.1, 0.2]
                    },
                    'linear_regression': {
                        'fit_intercept': [True, False]
                    }
                }
            
            return param_grids
            
        except Exception as e:
            logger.error(f"Erreur création grille hyperparamètres: {e}")
            raise
    
    def train_classification_models(self, train_data, test_data, feature_cols):
        """Entraîner et évaluer les modèles de classification"""
        try:
            logger.info("Entraînement des modèles de classification")
            
            # Séparer features et cible
            X_train = train_data[feature_cols]
            y_train = train_data['rating_category']
            X_test = test_data[feature_cols]
            y_test = test_data['rating_category']
            
            # Standardiser les features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Créer les modèles
            models = self.create_classification_models()
            
            # Créer les grilles d'hyperparamètres
            param_grids = self.create_hyperparameter_grid('classification')
            
            results = {}
            
            for model_name, model in models.items():
                logger.info(f"Entraînement: {model_name}")
                
                try:
                    # Créer le pipeline avec scaling
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('model', model)
                    ])
                    
                    # Optimisation des hyperparamètres
                    if model_name in param_grids:
                        param_grid = param_grids[model_name]
                        
                        # Grid Search avec Cross-Validation
                        grid_search = GridSearchCV(
                            pipeline,
                            param_grid,
                            cv=3,
                            scoring='roc_auc',
                            n_jobs=-1,
                            verbose=1
                        )
                        
                        # Entraîner avec grid search
                        grid_search.fit(X_train, y_train)
                        
                        # Meilleur modèle
                        best_model = grid_search.best_estimator_
                        best_params = grid_search.best_params_
                        
                        # Prédictions
                        y_pred = best_model.predict(X_test)
                        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                        
                        # Évaluation
                        auc = roc_auc_score(y_test, y_pred_proba)
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        results[model_name] = {
                            'model': best_model,
                            'auc': auc,
                            'accuracy': accuracy,
                            'best_params': best_params,
                            'predictions': y_pred,
                            'probabilities': y_pred_proba
                        }
                        
                    else:
                        # Entraînement simple
                        pipeline.fit(X_train, y_train)
                        
                        # Prédictions
                        y_pred = pipeline.predict(X_test)
                        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                        
                        # Évaluation
                        auc = roc_auc_score(y_test, y_pred_proba)
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        results[model_name] = {
                            'model': pipeline,
                            'auc': auc,
                            'accuracy': accuracy,
                            'best_params': {},
                            'predictions': y_pred,
                            'probabilities': y_pred_proba
                        }
                    
                    logger.info(f"{model_name}: AUC = {auc:.4f}, Accuracy = {accuracy:.4f}")
                    
                except Exception as e:
                    logger.error(f"Erreur entraînement {model_name}: {e}")
                    results[model_name] = {
                        'error': str(e),
                        'auc': 0,
                        'accuracy': 0
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur entraînement classification: {e}")
            raise
    
    def train_regression_models(self, train_data, test_data, feature_cols):
        """Entraîner et évaluer les modèles de régression"""
        try:
            logger.info("Entraînement des modèles de régression")
            
            # Séparer features et cible
            X_train = train_data[feature_cols]
            y_train = train_data['overall']
            X_test = test_data[feature_cols]
            y_test = test_data['overall']
            
            # Standardiser les features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Créer les modèles
            models = self.create_regression_models()
            
            # Créer les grilles d'hyperparamètres
            param_grids = self.create_hyperparameter_grid('regression')
            
            results = {}
            
            for model_name, model in models.items():
                logger.info(f"Entraînement: {model_name}")
                
                try:
                    # Créer le pipeline avec scaling
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('model', model)
                    ])
                    
                    # Optimisation des hyperparamètres
                    if model_name in param_grids:
                        param_grid = param_grids[model_name]
                        
                        # Grid Search avec Cross-Validation
                        grid_search = GridSearchCV(
                            pipeline,
                            param_grid,
                            cv=3,
                            scoring='neg_mean_squared_error',
                            n_jobs=-1,
                            verbose=1
                        )
                        
                        # Entraîner avec grid search
                        grid_search.fit(X_train, y_train)
                        
                        # Meilleur modèle
                        best_model = grid_search.best_estimator_
                        best_params = grid_search.best_params_
                        
                        # Prédictions
                        y_pred = best_model.predict(X_test)
                        
                        # Évaluation
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        results[model_name] = {
                            'model': best_model,
                            'rmse': rmse,
                            'mae': mae,
                            'r2': r2,
                            'best_params': best_params,
                            'predictions': y_pred
                        }
                        
                    else:
                        # Entraînement simple
                        pipeline.fit(X_train, y_train)
                        
                        # Prédictions
                        y_pred = pipeline.predict(X_test)
                        
                        # Évaluation
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        results[model_name] = {
                            'model': pipeline,
                            'rmse': rmse,
                            'mae': mae,
                            'r2': r2,
                            'best_params': {},
                            'predictions': y_pred
                        }
                    
                    logger.info(f"{model_name}: RMSE = {rmse:.4f}, MAE = {mae:.4f}, R² = {r2:.4f}")
                    
                except Exception as e:
                    logger.error(f"Erreur entraînement {model_name}: {e}")
                    results[model_name] = {
                        'error': str(e),
                        'rmse': float('inf'),
                        'mae': float('inf'),
                        'r2': 0
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur entraînement régression: {e}")
            raise
    
    def select_best_model(self, results, model_type='classification'):
        """Sélectionner le meilleur modèle"""
        try:
            logger.info(f"Sélection du meilleur modèle: {model_type}")
            
            # Filtrer les modèles avec erreurs
            valid_results = {k: v for k, v in results.items() if 'error' not in v}
            
            if not valid_results:
                return None
            
            if model_type == 'classification':
                # Sélectionner par AUC
                best_model_name = max(
                    valid_results.keys(),
                    key=lambda x: valid_results[x].get('auc', 0)
                )
                best_score = valid_results[best_model_name].get('auc', 0)
                metric_name = 'AUC'
                
            elif model_type == 'regression':
                # Sélectionner par RMSE (plus petit est meilleur)
                best_model_name = min(
                    valid_results.keys(),
                    key=lambda x: valid_results[x].get('rmse', float('inf'))
                )
                best_score = valid_results[best_model_name].get('rmse', float('inf'))
                metric_name = 'RMSE'
            
            best_model_info = valid_results[best_model_name]
            
            logger.info(f"Meilleur modèle: {best_model_name} ({metric_name} = {best_score:.4f})")
            
            return {
                'model_name': best_model_name,
                'model': best_model_info.get('model'),
                'score': best_score,
                'metric': metric_name,
                'params': best_model_info.get('best_params', {}),
                'all_results': results
            }
            
        except Exception as e:
            logger.error(f"Erreur sélection meilleur modèle: {e}")
            return None
    
    def save_model(self, model, model_name, model_path: str):
        """Sauvegarder le modèle entraîné"""
        try:
            logger.info(f"Sauvegarde du modèle: {model_name}")
            
            # Créer le répertoire
            import os
            os.makedirs(model_path, exist_ok=True)
            
            # Sauvegarder le modèle
            import joblib
            joblib.dump(model, f"{model_path}/{model_name}.pkl")
            
            logger.info(f"Modèle {model_name} sauvegardé dans: {model_path}")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde modèle: {e}")
            raise
    
    def create_evaluation_report(self, results, model_type, dataset_name):
        """Créer un rapport d'évaluation visuel"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'ML Evaluation - {model_type.title()} - {dataset_name}', fontsize=16)
            
            # 1. Comparaison des performances
            if model_type == 'classification':
                model_names = list(results.keys())
                auc_scores = [results[name].get('auc', 0) for name in model_names]
                accuracy_scores = [results[name].get('accuracy', 0) for name in model_names]
                
                x = np.arange(len(model_names))
                width = 0.35
                
                axes[0, 0].bar(x - width/2, auc_scores, width, label='AUC', alpha=0.8)
                axes[0, 0].bar(x + width/2, accuracy_scores, width, label='Accuracy', alpha=0.8)
                axes[0, 0].set_xlabel('Modèles')
                axes[0, 0].set_ylabel('Score')
                axes[0, 0].set_title('Performance Comparison')
                axes[0, 0].set_xticks(x)
                axes[0, 0].set_xticklabels(model_names, rotation=45)
                axes[0, 0].legend()
                
            elif model_type == 'regression':
                model_names = list(results.keys())
                rmse_scores = [results[name].get('rmse', 0) for name in model_names]
                mae_scores = [results[name].get('mae', 0) for name in model_names]
                
                x = np.arange(len(model_names))
                width = 0.35
                
                axes[0, 0].bar(x - width/2, rmse_scores, width, label='RMSE', alpha=0.8)
                axes[0, 0].bar(x + width/2, mae_scores, width, label='MAE', alpha=0.8)
                axes[0, 0].set_xlabel('Modèles')
                axes[0, 0].set_ylabel('Erreur')
                axes[0, 0].set_title('Performance Comparison')
                axes[0, 0].set_xticks(x)
                axes[0, 0].set_xticklabels(model_names, rotation=45)
                axes[0, 0].legend()
            
            # 2. Meilleur modèle
            best_model = self.select_best_model(results, model_type)
            if best_model:
                best_text = f"""Best Model: {best_model['model_name']}
Score: {best_model['score']:.4f}
Metric: {best_model['metric']}"""
                
                axes[0, 1].text(0.1, 0.5, best_text, ha='left', va='center',
                                  fontsize=12, transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Best Model')
                axes[0, 1].axis('off')
            
            # 3. Distribution des prédictions
            if results:
                sample_model = list(results.values())[0]
                if 'predictions' in sample_model:
                    predictions = sample_model['predictions']
                    
                    if model_type == 'classification':
                        pd.Series(predictions).value_counts().plot(kind='bar', ax=axes[1, 0], alpha=0.8)
                        axes[1, 0].set_title('Predictions Distribution')
                        axes[1, 0].set_xlabel('Predicted Class')
                        axes[1, 0].set_ylabel('Count')
                    else:
                        axes[1, 0].hist(predictions, bins=50, alpha=0.7)
                        axes[1, 0].set_title('Predictions Distribution')
                        axes[1, 0].set_xlabel('Predicted Rating')
                        axes[1, 0].set_ylabel('Frequency')
            
            # 4. Informations sur les hyperparamètres
            if best_model and best_model['params']:
                params_text = "Best Hyperparameters:\n"
                for param, value in best_model['params'].items():
                    params_text += f"{param}: {value}\n"
                
                axes[1, 1].text(0.1, 0.5, params_text, ha='left', va='top',
                                  fontsize=10, transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Best Hyperparameters')
                axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # Sauvegarder le rapport
            report_path = f"data/processed/ml_evaluation_{model_type}_{dataset_name.lower()}.png"
            plt.savefig(report_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"Rapport d'évaluation sauvegardé: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Erreur création rapport: {e}")
            return None
    
    def run_ml_pipeline(self, feature_store_path: str, model_output_path: str):
        """Exécuter le pipeline ML complet"""
        try:
            start_time = datetime.now()
            logger.info("Démarrage pipeline ML avancé")
            
            # 1. Charger les données
            df, feature_cols = self.load_feature_store(feature_store_path)
            
            # 2. Créer la cible de classification
            df = self.create_classification_target(df)
            
            # 3. Préparer les features
            df, feature_cols_final = self.prepare_features(df, feature_cols)
            
            # 4. Split train/test
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            
            # 5. Entraîner les modèles de classification
            classification_results = self.train_classification_models(train_df, test_df, feature_cols_final)
            
            # 6. Entraîner les modèles de régression
            regression_results = self.train_regression_models(train_df, test_df, feature_cols_final)
            
            # 7. Sélectionner les meilleurs modèles
            best_classification = self.select_best_model(classification_results, 'classification')
            best_regression = self.select_best_model(regression_results, 'regression')
            
            # 8. Sauvegarder les modèles
            if best_classification:
                self.save_model(
                    best_classification['model'], 
                    best_classification['model_name'], 
                    model_output_path
                )
            
            if best_regression:
                self.save_model(
                    best_regression['model'],
                    best_regression['model_name'],
                    model_output_path
                )
            
            # 9. Créer les rapports d'évaluation
            dataset_name = feature_store_path.split('/')[-1].replace('_feature_store.csv', '')
            
            if classification_results:
                self.create_evaluation_report(classification_results, 'classification', dataset_name)
            
            if regression_results:
                self.create_evaluation_report(regression_results, 'regression', dataset_name)
            
            # Finaliser les résultats
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            results = {
                'status': 'success',
                'duration_seconds': duration,
                'best_classification': best_classification,
                'best_regression': best_regression,
                'classification_results': classification_results,
                'regression_results': regression_results,
                'data_stats': {
                    'total_records': len(df),
                    'train_records': len(train_df),
                    'test_records': len(test_df),
                    'feature_count': len(feature_cols_final)
                }
            }
            
            logger.info(f"Pipeline ML terminé en {duration:.2f}s")
            
            if best_classification:
                logger.info(f"Meilleur classification: {best_classification['model_name']} ({best_classification['metric']} = {best_classification['score']:.4f})")
            
            if best_regression:
                logger.info(f"Meilleur régression: {best_regression['model_name']} ({best_regression['metric']} = {best_regression['score']:.4f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur pipeline ML: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

def main():
    """Fonction principale du ML avancé"""
    ml_pipeline = AdvancedMLSklearn()
    
    # Configuration des datasets
    datasets = [
        {
            'feature_store_path': "data/processed/electronics_features_feature_store.csv",
            'model_output_path': "models/ml_electronics",
            'name': 'Electronics'
        },
        {
            'feature_store_path': "data/processed/clothing_features_feature_store.csv",
            'model_output_path': "models/ml_clothing",
            'name': 'Clothing'
        }
    ]
    
    results = []
    
    # Exécuter pour chaque dataset
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"ML Avancé: {dataset['name']}")
        print(f"{'='*60}")
        
        result = ml_pipeline.run_ml_pipeline(
            dataset['feature_store_path'],
            dataset['model_output_path']
        )
        
        results.append(result)
        
        if result['status'] == 'success':
            print(f"✅ {dataset['name']}: Pipeline terminé")
            
            if result['best_classification']:
                best_clf = result['best_classification']
                print(f"   Meilleur classification: {best_clf['model_name']} ({best_clf['metric']} = {best_clf['score']:.4f})")
            
            if result['best_regression']:
                best_reg = result['best_regression']
                print(f"   Meilleur régression: {best_reg['model_name']} ({best_reg['metric']} = {best_reg['score']:.4f})")
            
            print(f"   Duration: {result['duration_seconds']:.2f}s")
        else:
            print(f"❌ {dataset['name']}: {result['error']}")
    
    # Rapport final
    print(f"\n{'='*60}")
    print("RAPPORT ML AVANCÉ FINAL")
    print(f"{'='*60}")
    
    successful_pipelines = [r for r in results if r['status'] == 'success']
    
    if successful_pipelines:
        print(f"Pipelines réussis: {len(successful_pipelines)}/{len(results)}")
        
        # Moyennes des performances
        clf_aucs = [r['best_classification']['score'] for r in successful_pipelines if r['best_classification']]
        reg_rmses = [r['best_regression']['score'] for r in successful_pipelines if r['best_regression']]
        
        if clf_aucs:
            avg_auc = np.mean(clf_aucs)
            print(f"AUC moyen: {avg_auc:.4f}")
            
            if avg_auc > 0.8:
                print("✅ Objectif AUC > 0.8 atteint!")
            else:
                print("⚠️ Objectif AUC > 0.8 non atteint")
        
        if reg_rmses:
            avg_rmse = np.mean(reg_rmses)
            print(f"RMSE moyen: {avg_rmse:.4f}")
            
            if avg_rmse < 1.0:
                print("✅ Objectif RMSE < 1.0 atteint!")
            else:
                print("⚠️ Objectif RMSE < 1.0 non atteint")
    else:
        print("❌ Aucun pipeline réussi")
    
    return results

if __name__ == "__main__":
    main()
