"""
MLlib avancé : Pipelines ML distribués et hyperparameter tuning
PySpark ML pipelines avec cross-validation et optimisation
"""

import sys
import os
sys.path.append('src')

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, GBTClassifier
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor, LinearRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
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

class AdvancedMLlib:
    """Pipeline ML avancé avec PySpark MLlib"""
    
    def __init__(self):
        """Initialiser le pipeline ML"""
        self.spark = None
        self.pipelines = {}
        self.models = {}
        self.evaluation_results = {}
        self.best_model = None
        self.best_pipeline = None
        
    def initialize_spark(self):
        """Initialiser la session Spark"""
        try:
            self.spark = SparkSession.builder \
                .appName("mllib_advanced") \
                .master("local[*]") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                .getOrCreate()
            
            logger.info("Session Spark initialisée pour MLlib")
            return self.spark
            
        except Exception as e:
            logger.error(f"Erreur initialisation Spark: {e}")
            raise
    
    def load_feature_store(self, csv_path: str):
        """Charger les données du feature store"""
        try:
            logger.info(f"Chargement des données depuis: {csv_path}")
            
            # Charger avec Pandas puis convertir en Spark
            df_pandas = pd.read_csv(csv_path)
            
            # Sélectionner les features pertinentes
            feature_cols = [
                'overall', 'word_count', 'sentiment_score', 'helpfulness_ratio',
                'user_review_count', 'user_avg_rating', 'product_review_count', 
                'product_avg_rating', 'user_experience', 'product_popularity'
            ]
            
            # Vérifier les colonnes disponibles
            available_cols = [col for col in feature_cols if col in df_pandas.columns]
            df_pandas = df_pandas[available_cols].copy()
            
            # Nettoyer les données
            df_pandas = df_pandas.dropna()
            
            # Convertir en Spark DataFrame
            df_spark = self.spark.createDataFrame(df_pandas)
            
            logger.info(f"Données chargées: {df_spark.count():,} enregistrements")
            logger.info(f"Features disponibles: {available_cols}")
            
            return df_spark, available_cols
            
        except Exception as e:
            logger.error(f"Erreur chargement données: {e}")
            raise
    
    def create_classification_target(self, df):
        """Créer la cible de classification (rating positif/négatif)"""
        try:
            logger.info("Création de la cible de classification")
            
            # Créer la cible binaire
            df = df.withColumn(
                'rating_category',
                F.when(F.col('overall') >= 4, 1).otherwise(0)
            )
            
            # Statistiques de la cible
            positive_count = df.filter(F.col('rating_category') == 1).count()
            negative_count = df.filter(F.col('rating_category') == 0).count()
            total_count = df.count()
            
            logger.info(f"Ratings positifs: {positive_count} ({positive_count/total_count:.1%})")
            logger.info(f"Ratings négatifs: {negative_count} ({negative_count/total_count:.1%})")
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur création cible: {e}")
            raise
    
    def create_features_pipeline(self, feature_cols):
        """Créer le pipeline de preprocessing des features"""
        try:
            logger.info("Création du pipeline de features")
            
            # Indexer les colonnes catégorielles
            categorical_cols = ['user_experience', 'product_popularity']
            categorical_cols = [col for col in categorical_cols if col in feature_cols]
            
            indexers = []
            for col in categorical_cols:
                indexer = StringIndexer(inputCol=col, outputCol=f"{col}_indexed")
                indexers.append(indexer)
            
            # Encoder les colonnes catégorielles
            encoders = []
            for col in categorical_cols:
                encoder = OneHotEncoder(inputCol=f"{col}_indexed", outputCol=f"{col}_encoded")
                encoders.append(encoder)
            
            # Sélectionner les features numériques
            numeric_cols = [col for col in feature_cols if col not in categorical_cols]
            
            # Assembler toutes les features
            encoded_cols = [f"{col}_encoded" for col in categorical_cols]
            all_feature_cols = numeric_cols + encoded_cols
            
            assembler = VectorAssembler(inputCols=all_feature_cols, outputCol="features")
            
            # Standardiser les features
            scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
            
            # Créer le pipeline complet
            pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler])
            
            logger.info(f"Pipeline créé avec {len(feature_cols)} features")
            return pipeline, "scaled_features"
            
        except Exception as e:
            logger.error(f"Erreur création pipeline: {e}")
            raise
    
    def create_classification_models(self):
        """Créer les modèles de classification"""
        try:
            logger.info("Création des modèles de classification")
            
            models = {
                'logistic_regression': LogisticRegression(
                    featuresCol="scaled_features",
                    labelCol="rating_category",
                    maxIter=100,
                    regParam=0.1,
                    elasticNetParam=0.8
                ),
                'random_forest': RandomForestClassifier(
                    featuresCol="scaled_features",
                    labelCol="rating_category",
                    numTrees=100,
                    maxDepth=10,
                    minInstancesPerNode=5
                ),
                'gradient_boosting': GBTClassifier(
                    featuresCol="scaled_features",
                    labelCol="rating_category",
                    maxIter=100,
                    maxDepth=8,
                    stepSize=0.1
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
                    featuresCol="scaled_features",
                    labelCol="overall",
                    maxIter=100,
                    regParam=0.1,
                    elasticNetParam=0.8
                ),
                'random_forest_reg': RandomForestRegressor(
                    featuresCol="scaled_features",
                    labelCol="overall",
                    numTrees=100,
                    maxDepth=10,
                    minInstancesPerNode=5
                ),
                'gradient_boosting_reg': GBTRegressor(
                    featuresCol="scaled_features",
                    labelCol="overall",
                    maxIter=100,
                    maxDepth=8,
                    stepSize=0.1
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
                # Grille pour Random Forest
                rf_param_grid = ParamGridBuilder() \
                    .addGrid(RandomForestClassifier.numTrees, [50, 100, 200]) \
                    .addGrid(RandomForestClassifier.maxDepth, [5, 10, 15]) \
                    .addGrid(RandomForestClassifier.minInstancesPerNode, [1, 5, 10]) \
                    .build()
                
                # Grille pour Gradient Boosting
                gbt_param_grid = ParamGridBuilder() \
                    .addGrid(GBTClassifier.maxIter, [50, 100, 200]) \
                    .addGrid(GBTClassifier.maxDepth, [5, 8, 12]) \
                    .addGrid(GBTClassifier.stepSize, [0.05, 0.1, 0.2]) \
                    .build()
                
                return {
                    'random_forest': rf_param_grid,
                    'gradient_boosting': gbt_param_grid
                }
                
            elif model_type == 'regression':
                # Grille pour Random Forest Regressor
                rf_reg_param_grid = ParamGridBuilder() \
                    .addGrid(RandomForestRegressor.numTrees, [50, 100, 200]) \
                    .addGrid(RandomForestRegressor.maxDepth, [5, 10, 15]) \
                    .addGrid(RandomForestRegressor.minInstancesPerNode, [1, 5, 10]) \
                    .build()
                
                # Grille pour Gradient Boosting Regressor
                gbt_reg_param_grid = ParamGridBuilder() \
                    .addGrid(GBTRegressor.maxIter, [50, 100, 200]) \
                    .addGrid(GBTRegressor.maxDepth, [5, 8, 12]) \
                    .addGrid(GBTRegressor.stepSize, [0.05, 0.1, 0.2]) \
                    .build()
                
                return {
                    'random_forest_reg': rf_reg_param_grid,
                    'gradient_boosting_reg': gbt_reg_param_grid
                }
                
        except Exception as e:
            logger.error(f"Erreur création grille hyperparamètres: {e}")
            raise
    
    def train_classification_models(self, train_data, test_data, feature_cols):
        """Entraîner et évaluer les modèles de classification"""
        try:
            logger.info("Entraînement des modèles de classification")
            
            # Créer la cible
            train_data = self.create_classification_target(train_data)
            test_data = self.create_classification_target(test_data)
            
            # Créer le pipeline de features
            feature_pipeline, features_col = self.create_features_pipeline(feature_cols)
            
            # Créer les modèles
            models = self.create_classification_models()
            
            # Créer les grilles d'hyperparamètres
            param_grids = self.create_hyperparameter_grid('classification')
            
            # Évaluateur
            evaluator = BinaryClassificationEvaluator(
                labelCol="rating_category",
                rawPredictionCol="rawPrediction",
                metricName="areaUnderROC"
            )
            
            results = {}
            
            for model_name, model in models.items():
                logger.info(f"Entraînement: {model_name}")
                
                try:
                    # Créer le pipeline complet
                    full_pipeline = Pipeline(stages=[feature_pipeline, model])
                    
                    # Optimisation des hyperparamètres
                    if model_name in param_grids:
                        param_grid = param_grids[model_name]
                        
                        # Cross-validation
                        crossval = CrossValidator(
                            estimator=full_pipeline,
                            estimatorParamMaps=param_grid,
                            evaluator=evaluator,
                            numFolds=3,
                            parallelism=2
                        )
                        
                        # Entraîner avec cross-validation
                        cv_model = crossval.fit(train_data)
                        
                        # Meilleur modèle
                        best_model = cv_model.bestModel
                        best_params = cv_model.getEstimatorParamMaps()[cv_model.bestModelIndex]
                        
                        # Prédictions
                        predictions = best_model.transform(test_data)
                        
                        # Évaluation
                        auc = evaluator.evaluate(predictions)
                        accuracy = predictions.filter(
                            predictions.rating_category == predictions.prediction
                        ).count() / predictions.count()
                        
                        results[model_name] = {
                            'model': best_model,
                            'auc': auc,
                            'accuracy': accuracy,
                            'best_params': best_params,
                            'predictions': predictions
                        }
                        
                    else:
                        # Entraînement simple
                        full_pipeline = Pipeline(stages=[feature_pipeline, model])
                        trained_model = full_pipeline.fit(train_data)
                        
                        # Prédictions
                        predictions = trained_model.transform(test_data)
                        
                        # Évaluation
                        auc = evaluator.evaluate(predictions)
                        accuracy = predictions.filter(
                            predictions.rating_category == predictions.prediction
                        ).count() / predictions.count()
                        
                        results[model_name] = {
                            'model': trained_model,
                            'auc': auc,
                            'accuracy': accuracy,
                            'best_params': {},
                            'predictions': predictions
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
            
            # Créer le pipeline de features
            feature_pipeline, features_col = self.create_features_pipeline(feature_cols)
            
            # Créer les modèles
            models = self.create_regression_models()
            
            # Créer les grilles d'hyperparamètres
            param_grids = self.create_hyperparameter_grid('regression')
            
            # Évaluateur
            evaluator = RegressionEvaluator(
                labelCol="overall",
                predictionCol="prediction",
                metricName="rmse"
            )
            
            results = {}
            
            for model_name, model in models.items():
                logger.info(f"Entraînement: {model_name}")
                
                try:
                    # Créer le pipeline complet
                    full_pipeline = Pipeline(stages=[feature_pipeline, model])
                    
                    # Optimisation des hyperparamètres
                    if model_name in param_grids:
                        param_grid = param_grids[model_name]
                        
                        # Cross-validation
                        crossval = CrossValidator(
                            estimator=full_pipeline,
                            estimatorParamMaps=param_grid,
                            evaluator=evaluator,
                            numFolds=3,
                            parallelism=2
                        )
                        
                        # Entraîner avec cross-validation
                        cv_model = crossval.fit(train_data)
                        
                        # Meilleur modèle
                        best_model = cv_model.bestModel
                        best_params = cv_model.getEstimatorParamMaps()[cv_model.bestModelIndex]
                        
                        # Prédictions
                        predictions = best_model.transform(test_data)
                        
                        # Évaluation
                        rmse = evaluator.evaluate(predictions)
                        mae = RegressionEvaluator(
                            labelCol="overall",
                            predictionCol="prediction",
                            metricName="mae"
                        ).evaluate(predictions)
                        
                        r2 = RegressionEvaluator(
                            labelCol="overall",
                            predictionCol="prediction",
                            metricName="r2"
                        ).evaluate(predictions)
                        
                        results[model_name] = {
                            'model': best_model,
                            'rmse': rmse,
                            'mae': mae,
                            'r2': r2,
                            'best_params': best_params,
                            'predictions': predictions
                        }
                        
                    else:
                        # Entraînement simple
                        full_pipeline = Pipeline(stages=[feature_pipeline, model])
                        trained_model = full_pipeline.fit(train_data)
                        
                        # Prédictions
                        predictions = trained_model.transform(test_data)
                        
                        # Évaluation
                        rmse = evaluator.evaluate(predictions)
                        mae = RegressionEvaluator(
                            labelCol="overall",
                            predictionCol="prediction",
                            metricName="mae"
                        ).evaluate(predictions)
                        
                        r2 = RegressionEvaluator(
                            labelCol="overall",
                            predictionCol="prediction",
                            metricName="r2"
                        ).evaluate(predictions)
                        
                        results[model_name] = {
                            'model': trained_model,
                            'rmse': rmse,
                            'mae': mae,
                            'r2': r2,
                            'best_params': {},
                            'predictions': predictions
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
            
            if model_type == 'classification':
                # Sélectionner par AUC
                best_model_name = max(
                    results.keys(),
                    key=lambda x: results[x].get('auc', 0)
                )
                best_score = results[best_model_name].get('auc', 0)
                metric_name = 'AUC'
                
            elif model_type == 'regression':
                # Sélectionner par RMSE (plus petit est meilleur)
                best_model_name = min(
                    results.keys(),
                    key=lambda x: results[x].get('rmse', float('inf'))
                )
                best_score = results[best_model_name].get('rmse', float('inf'))
                metric_name = 'RMSE'
            
            best_model_info = results[best_model_name]
            
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
            os.makedirs(model_path, exist_ok=True)
            
            # Sauvegarder le modèle
            model.write().overwrite().save(f"{model_path}/{model_name}")
            
            logger.info(f"Modèle {model_name} sauvegardé dans: {model_path}")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde modèle: {e}")
            raise
    
    def create_evaluation_report(self, results, model_type, dataset_name):
        """Créer un rapport d'évaluation visuel"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'MLlib Evaluation - {model_type.title()} - {dataset_name}', fontsize=16)
            
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
                if 'predictions' in sample_model and sample_model['predictions']:
                    predictions_df = sample_model['predictions'].toPandas()
                    
                    if model_type == 'classification':
                        predictions_df['prediction'].hist(bins=20, ax=axes[1, 0], alpha=0.7)
                        axes[1, 0].set_title('Predictions Distribution')
                        axes[1, 0].set_xlabel('Predicted Class')
                        axes[1, 0].set_ylabel('Frequency')
                    else:
                        predictions_df['prediction'].hist(bins=50, ax=axes[1, 0], alpha=0.7)
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
            report_path = f"data/processed/mllib_evaluation_{model_type}_{dataset_name.lower()}.png"
            plt.savefig(report_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"Rapport d'évaluation sauvegardé: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Erreur création rapport: {e}")
            return None
    
    def run_mllib_pipeline(self, feature_store_path: str, model_output_path: str):
        """Exécuter le pipeline MLlib complet"""
        try:
            start_time = datetime.now()
            logger.info("Démarrage pipeline MLlib avancé")
            
            # 1. Initialiser Spark
            self.initialize_spark()
            
            # 2. Charger les données
            df, feature_cols = self.load_feature_store(feature_store_path)
            
            # 3. Split train/test
            train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
            
            # 4. Entraîner les modèles de classification
            classification_results = self.train_classification_models(train_df, test_df, feature_cols)
            
            # 5. Entraîner les modèles de régression
            regression_results = self.train_regression_models(train_df, test_df, feature_cols)
            
            # 6. Sélectionner les meilleurs modèles
            best_classification = self.select_best_model(classification_results, 'classification')
            best_regression = self.select_best_model(regression_results, 'regression')
            
            # 7. Sauvegarder les modèles
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
            
            # 8. Créer les rapports d'évaluation
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
                    'total_records': df.count(),
                    'train_records': train_df.count(),
                    'test_records': test_df.count(),
                    'feature_count': len(feature_cols)
                }
            }
            
            logger.info(f"Pipeline MLlib terminé en {duration:.2f}s")
            
            if best_classification:
                logger.info(f"Meilleur classification: {best_classification['model_name']} ({best_classification['metric']} = {best_classification['score']:.4f})")
            
            if best_regression:
                logger.info(f"Meilleur régression: {best_regression['model_name']} ({best_regression['metric']} = {best_regression['score']:.4f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur pipeline MLlib: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
            
        finally:
            if self.spark:
                self.spark.stop()
                logger.info("Session Spark arrêtée")

def main():
    """Fonction principale du MLlib avancé"""
    mllib = AdvancedMLlib()
    
    # Configuration des datasets
    datasets = [
        {
            'feature_store_path': "data/processed/electronics_features_feature_store.csv",
            'model_output_path': "models/mllib_electronics",
            'name': 'Electronics'
        },
        {
            'feature_store_path': "data/processed/clothing_features_feature_store.csv",
            'model_output_path': "models/mllib_clothing",
            'name': 'Clothing'
        }
    ]
    
    results = []
    
    # Exécuter pour chaque dataset
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"MLlib Avancé: {dataset['name']}")
        print(f"{'='*60}")
        
        result = mllib.run_mllib_pipeline(
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
    print("RAPPORT MLLIB AVANCÉ FINAL")
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
