"""
Optimisation des performances et Model Selection
Analyse des modèles, benchmarking et sélection du meilleur modèle
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Optimisation et sélection de modèles"""
    
    def __init__(self):
        """Initialiser l'optimiseur"""
        self.models = {}
        self.benchmark_results = {}
        self.best_model = None
        self.feature_importance = {}
        
    def load_models(self, model_path: str):
        """Charger les modèles entraînés"""
        try:
            logger.info(f"Chargement des modèles depuis: {model_path}")
            
            import os
            model_files = [f for f in os.listdir(model_path) if f.endswith('.pkl')]
            
            for model_file in model_files:
                model_name = model_file.replace('.pkl', '')
                model = joblib.load(f"{model_path}/{model_file}")
                self.models[model_name] = model
                
                logger.info(f"Modèle chargé: {model_name}")
            
            return self.models
            
        except Exception as e:
            logger.error(f"Erreur chargement modèles: {e}")
            raise
    
    def benchmark_models(self, X_test, y_test):
        """Benchmark des performances des modèles"""
        try:
            logger.info("Benchmark des modèles")
            
            results = {}
            
            for model_name, model in self.models.items():
                try:
                    # Prédictions
                    y_pred = model.predict(X_test)
                    
                    # Métriques
                    if len(y_pred.shape) == 1:
                        # Classification
                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                        
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted')
                        recall = recall_score(y_test, y_pred, average='weighted')
                        f1 = f1_score(y_test, y_pred, average='weighted')
                        
                        results[model_name] = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'type': 'classification'
                        }
                        
                        logger.info(f"{model_name}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")
                        
                    else:
                        # Régression
                        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                        
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        results[model_name] = {
                            'rmse': rmse,
                            'mae': mae,
                            'r2': r2,
                            'type': 'regression'
                        }
                        
                        logger.info(f"{model_name}: RMSE = {rmse:.4f}, R² = {r2:.4f}")
                        
                except Exception as e:
                    logger.error(f"Erreur benchmark {model_name}: {e}")
                    results[model_name] = {'error': str(e)}
            
            self.benchmark_results = results
            return results
            
        except Exception as e:
            logger.error(f"Erreur benchmark: {e}")
            raise
    
    def select_best_model(self):
        """Sélectionner le meilleur modèle"""
        try:
            logger.info("Sélection du meilleur modèle")
            
            # Filtrer les modèles avec erreurs
            valid_results = {k: v for k, v in self.benchmark_results.items() if 'error' not in v}
            
            if not valid_results:
                return None
            
            # Sélectionner selon le type
            classification_models = {k: v for k, v in valid_results.items() if v.get('type') == 'classification'}
            regression_models = {k: v for k, v in valid_results.items() if v.get('type') == 'regression'}
            
            best_classification = None
            best_regression = None
            
            if classification_models:
                # Sélectionner par F1 score
                best_clf_name = max(classification_models.keys(), key=lambda x: classification_models[x].get('f1_score', 0))
                best_classification = {
                    'name': best_clf_name,
                    'metrics': classification_models[best_clf_name],
                    'model': self.models.get(best_clf_name)
                }
                logger.info(f"Meilleur classification: {best_clf_name} (F1 = {classification_models[best_clf_name]['f1_score']:.4f})")
            
            if regression_models:
                # Sélectionner par RMSE (plus petit est meilleur)
                best_reg_name = min(regression_models.keys(), key=lambda x: regression_models[x].get('rmse', float('inf')))
                best_regression = {
                    'name': best_reg_name,
                    'metrics': regression_models[best_reg_name],
                    'model': self.models.get(best_reg_name)
                }
                logger.info(f"Meilleur régression: {best_reg_name} (RMSE = {regression_models[best_reg_name]['rmse']:.4f})")
            
            return {
                'best_classification': best_classification,
                'best_regression': best_regression
            }
            
        except Exception as e:
            logger.error(f"Erreur sélection meilleur modèle: {e}")
            return None
    
    def analyze_feature_importance(self, model, feature_names):
        """Analyser l'importance des features"""
        try:
            logger.info("Analyse de l'importance des features")
            
            # Extraire le modèle du pipeline
            if hasattr(model, 'named_steps'):
                estimator = model.named_steps.get('model')
            else:
                estimator = model
            
            # Obtenir l'importance des features
            if hasattr(estimator, 'feature_importances_'):
                importance = estimator.feature_importances_
                
                # Créer un DataFrame
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                self.feature_importance = importance_df.to_dict('records')
                
                logger.info(f"Top 5 features: {importance_df.head(5)['feature'].tolist()}")
                
                return importance_df
                
            elif hasattr(estimator, 'coef_'):
                # Pour les modèles linéaires
                coef = np.abs(estimator.coef_[0])
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': coef
                }).sort_values('importance', ascending=False)
                
                self.feature_importance = importance_df.to_dict('records')
                
                logger.info(f"Top 5 features: {importance_df.head(5)['feature'].tolist()}")
                
                return importance_df
            else:
                logger.warning("Le modèle ne supporte pas l'analyse d'importance")
                return None
                
        except Exception as e:
            logger.error(f"Erreur analyse importance: {e}")
            return None
    
    def create_learning_curves(self, model, X, y, model_name):
        """Créer les courbes d'apprentissage"""
        try:
            logger.info(f"Création des courbes d'apprentissage: {model_name}")
            
            # Créer les courbes d'apprentissage
            train_sizes, train_scores, val_scores = learning_curve(
                model, X, y,
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=5,
                n_jobs=-1
            )
            
            # Calculer les moyennes et écarts-types
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            # Créer le graphique
            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training Score')
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
            plt.plot(train_sizes, val_mean, 'o-', color='g', label='Validation Score')
            plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='g')
            
            plt.xlabel('Training Set Size')
            plt.ylabel('Score')
            plt.title(f'Learning Curves - {model_name}')
            plt.legend(loc='best')
            plt.grid(True)
            
            # Sauvegarder
            plt.savefig(f"data/processed/learning_curves_{model_name}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Courbes d'apprentissage sauvegardées")
            
            return {
                'train_sizes': train_sizes.tolist(),
                'train_mean': train_mean.tolist(),
                'val_mean': val_mean.tolist()
            }
            
        except Exception as e:
            logger.error(f"Erreur création courbes: {e}")
            return None
    
    def create_confusion_matrix(self, model, X_test, y_test, model_name):
        """Créer la matrice de confusion"""
        try:
            logger.info(f"Création matrice de confusion: {model_name}")
            
            # Prédictions
            y_pred = model.predict(X_test)
            
            # Matrice de confusion
            cm = confusion_matrix(y_test, y_pred)
            
            # Créer le graphique
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix - {model_name}')
            
            # Sauvegarder
            plt.savefig(f"data/processed/confusion_matrix_{model_name}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Matrice de confusion sauvegardée")
            
            return cm.tolist()
            
        except Exception as e:
            logger.error(f"Erreur création matrice confusion: {e}")
            return None
    
    def create_benchmark_report(self, dataset_name):
        """Créer un rapport de benchmark visuel"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Model Benchmark - {dataset_name}', fontsize=16)
            
            # 1. Comparaison des performances (Classification)
            classification_results = {
                k: v for k, v in self.benchmark_results.items() 
                if v.get('type') == 'classification' and 'error' not in v
            }
            
            if classification_results:
                model_names = list(classification_results.keys())
                metrics = ['accuracy', 'precision', 'recall', 'f1_score']
                
                x = np.arange(len(model_names))
                width = 0.2
                
                for i, metric in enumerate(metrics):
                    values = [classification_results[name].get(metric, 0) for name in model_names]
                    axes[0, 0].bar(x + i*width, values, width, label=metric, alpha=0.8)
                
                axes[0, 0].set_xlabel('Modèles')
                axes[0, 0].set_ylabel('Score')
                axes[0, 0].set_title('Classification Performance')
                axes[0, 0].set_xticks(x + width)
                axes[0, 0].set_xticklabels(model_names, rotation=45)
                axes[0, 0].legend()
            
            # 2. Comparaison des performances (Régression)
            regression_results = {
                k: v for k, v in self.benchmark_results.items() 
                if v.get('type') == 'regression' and 'error' not in v
            }
            
            if regression_results:
                model_names = list(regression_results.keys())
                metrics = ['rmse', 'mae', 'r2']
                
                x = np.arange(len(model_names))
                width = 0.25
                
                for i, metric in enumerate(metrics):
                    values = [regression_results[name].get(metric, 0) for name in model_names]
                    axes[0, 1].bar(x + i*width, values, width, label=metric, alpha=0.8)
                
                axes[0, 1].set_xlabel('Modèles')
                axes[0, 1].set_ylabel('Score')
                axes[0, 1].set_title('Regression Performance')
                axes[0, 1].set_xticks(x + width)
                axes[0, 1].set_xticklabels(model_names, rotation=45)
                axes[0, 1].legend()
            
            # 3. Meilleur modèle
            best_models = self.select_best_model()
            if best_models:
                if best_models['best_classification']:
                    best_clf = best_models['best_classification']
                    clf_text = f"""Best Classification: {best_clf['name']}
F1 Score: {best_clf['metrics']['f1_score']:.4f}
Accuracy: {best_clf['metrics']['accuracy']:.4f}"""
                    axes[1, 0].text(0.1, 0.5, clf_text, ha='left', va='center',
                                      fontsize=12, transform=axes[1, 0].transAxes)
                
                if best_models['best_regression']:
                    best_reg = best_models['best_regression']
                    reg_text = f"""Best Regression: {best_reg['name']}
RMSE: {best_reg['metrics']['rmse']:.4f}
R²: {best_reg['metrics']['r2']:.4f}"""
                    axes[1, 0].text(0.5, 0.5, reg_text, ha='left', va='center',
                                      fontsize=12, transform=axes[1, 0].transAxes)
                
                axes[1, 0].set_title('Best Models')
                axes[1, 0].axis('off')
            
            # 4. Feature importance
            if self.feature_importance:
                importance_df = pd.DataFrame(self.feature_importance)
                top_features = importance_df.head(10)
                
                axes[1, 1].barh(top_features['feature'], top_features['importance'])
                axes[1, 1].set_xlabel('Importance')
                axes[1, 1].set_title('Top 10 Feature Importance')
                axes[1, 1].invert_yaxis()
            
            plt.tight_layout()
            
            # Sauvegarder
            report_path = f"data/processed/benchmark_report_{dataset_name.lower()}.png"
            plt.savefig(report_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Rapport de benchmark sauvegardé: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Erreur création rapport benchmark: {e}")
            return None
    
    def run_optimization_pipeline(self, model_path: str, X_test, y_test, feature_names, dataset_name):
        """Exécuter le pipeline d'optimisation"""
        try:
            start_time = datetime.now()
            logger.info("Démarrage pipeline d'optimisation")
            
            # 1. Charger les modèles
            self.load_models(model_path)
            
            # 2. Benchmark des modèles
            benchmark_results = self.benchmark_models(X_test, y_test)
            
            # 3. Sélectionner le meilleur modèle
            best_models = self.select_best_model()
            
            # 4. Analyser l'importance des features
            if best_models and best_models['best_classification']:
                self.analyze_feature_importance(
                    best_models['best_classification']['model'],
                    feature_names
                )
            
            # 5. Créer les courbes d'apprentissage
            if best_models and best_models['best_classification']:
                self.create_learning_curves(
                    best_models['best_classification']['model'],
                    X_test,
                    y_test,
                    best_models['best_classification']['name']
                )
            
            # 6. Créer la matrice de confusion
            if best_models and best_models['best_classification']:
                self.create_confusion_matrix(
                    best_models['best_classification']['model'],
                    X_test,
                    y_test,
                    best_models['best_classification']['name']
                )
            
            # 7. Créer le rapport de benchmark
            self.create_benchmark_report(dataset_name)
            
            # Finaliser les résultats
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            results = {
                'status': 'success',
                'duration_seconds': duration,
                'benchmark_results': benchmark_results,
                'best_models': best_models,
                'feature_importance': self.feature_importance
            }
            
            logger.info(f"Pipeline d'optimisation terminé en {duration:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur pipeline optimisation: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

def main():
    """Fonction principale"""
    optimizer = ModelOptimizer()
    
    # Charger les données de test
    import sys
    sys.path.append('.')
    from src.models.mllib_sklearn import AdvancedMLSklearn
    ml_pipeline = AdvancedMLSklearn()
    
    # Configuration des datasets
    datasets = [
        {
            'feature_store_path': "data/processed/electronics_features_feature_store.csv",
            'model_path': "models/ml_electronics",
            'name': 'Electronics'
        },
        {
            'feature_store_path': "data/processed/clothing_features_feature_store.csv",
            'model_path': "models/ml_clothing",
            'name': 'Clothing'
        }
    ]
    
    results = []
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Optimisation: {dataset['name']}")
        print(f"{'='*60}")
        
        try:
            # Charger les données
            df, feature_cols = ml_pipeline.load_feature_store(dataset['feature_store_path'])
            df = ml_pipeline.create_classification_target(df)
            df, feature_cols_final = ml_pipeline.prepare_features(df, feature_cols)
            
            # Split train/test
            from sklearn.model_selection import train_test_split
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            
            X_test = test_df[feature_cols_final]
            y_test = test_df['rating_category']
            
            # Exécuter l'optimisation
            result = optimizer.run_optimization_pipeline(
                dataset['model_path'],
                X_test,
                y_test,
                feature_cols_final,
                dataset['name']
            )
            
            results.append(result)
            
            if result['status'] == 'success':
                print(f"✅ {dataset['name']}: Optimisation terminée")
                print(f"   Duration: {result['duration_seconds']:.2f}s")
            else:
                print(f"❌ {dataset['name']}: {result['error']}")
                
        except Exception as e:
            print(f"❌ {dataset['name']}: {str(e)}")
            results.append({'status': 'error', 'error': str(e)})
    
    # Rapport final
    print(f"\n{'='*60}")
    print("RAPPORT OPTIMISATION FINAL")
    print(f"{'='*60}")
    
    successful_optimizations = [r for r in results if r['status'] == 'success']
    
    if successful_optimizations:
        print(f"Optimisations réussies: {len(successful_optimizations)}/{len(results)}")
    else:
        print("❌ Aucune optimisation réussie")
    
    return results

if __name__ == "__main__":
    main()
