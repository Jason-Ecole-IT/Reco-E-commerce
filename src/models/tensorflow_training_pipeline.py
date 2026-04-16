"""
Pipeline de Training et Évaluation TensorFlow
Training complet avec validation et métriques
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensorFlowTrainingPipeline:
    """Pipeline de training TensorFlow complet"""
    
    def __init__(self):
        """Initialiser le pipeline"""
        self.models = {}
        self.histories = {}
        self.evaluation_results = {}
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Charger les données du feature store"""
        try:
            logger.info(f"Chargement des données depuis: {data_path}")
            
            df = pd.read_csv(data_path)
            
            # Nettoyer les données
            df = df.dropna()
            
            logger.info(f"Données chargées: {len(df)} enregistrements")
            return df
            
        except Exception as e:
            logger.error(f"Erreur chargement données: {e}")
            raise
    
    def prepare_ncf_data(self, df: pd.DataFrame) -> Tuple:
        """
        Préparer les données pour NCF
        
        Returns:
            (user_indices, item_indices, targets, user_mapping, item_mapping)
        """
        try:
            # Créer les mappings
            unique_users = sorted(df['reviewerID'].unique())
            unique_items = sorted(df['asin'].unique())
            
            user_mapping = {user: idx for idx, user in enumerate(unique_users)}
            item_mapping = {item: idx for idx, item in enumerate(unique_items)}
            
            reverse_user_mapping = {idx: user for user, idx in user_mapping.items()}
            reverse_item_mapping = {idx: item for item, idx in item_mapping.items()}
            
            # Convertir en indices
            df['user_idx'] = df['reviewerID'].map(user_mapping)
            df['item_idx'] = df['asin'].map(item_mapping)
            
            # Créer target binaire
            df['target'] = (df['overall'] >= 4).astype(int)
            
            # Extraire les arrays
            user_indices = df['user_idx'].values
            item_indices = df['item_idx'].values
            targets = df['target'].values
            
            num_users = len(unique_users)
            num_items = len(unique_items)
            
            logger.info(f"Données NCF préparées: {num_users} users, {num_items} items")
            
            return (user_indices, item_indices, targets, 
                   user_mapping, item_mapping, 
                   reverse_user_mapping, reverse_item_mapping,
                   num_users, num_items)
            
        except Exception as e:
            logger.error(f"Erreur préparation données NCF: {e}")
            raise
    
    def train_ncf_model(self, user_indices: np.ndarray, item_indices: np.ndarray, 
                        targets: np.ndarray, num_users: int, num_items: int,
                        embedding_dim: int = 64, hidden_units: List[int] = [128, 64, 32],
                        epochs: int = 10, batch_size: int = 256, validation_split: float = 0.2):
        """
        Entraîner le modèle NCF
        
        Args:
            user_indices: Indices utilisateurs
            item_indices: Indices items
            targets: Targets binaires
            num_users: Nombre d'utilisateurs
            num_items: Nombre d'items
            embedding_dim: Dimension des embeddings
            hidden_units: Unités cachées MLP
            epochs: Nombre d'epochs
            batch_size: Taille de batch
            validation_split: Ratio validation
        """
        try:
            from src.models.neural_collaborative_filtering import NCFModel
            
            # Créer le modèle
            ncf = NCFModel(num_users=num_users, num_items=num_items, embedding_dim=embedding_dim)
            ncf.build_model(hidden_units=hidden_units)
            
            # Callbacks
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
            
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
            
            model_checkpoint = callbacks.ModelCheckpoint(
                'models/ncf_best_model.h5',
                monitor='val_loss',
                save_best_only=True
            )
            
            # Training
            logger.info("Début entraînement NCF...")
            history = ncf.train(
                user_indices, item_indices, targets,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Sauvegarder le modèle
            ncf.save_model("models/ncf_trained")
            
            # Sauvegarder l'historique
            self.histories['ncf'] = history.history
            
            logger.info("Entraînement NCF terminé")
            
            return ncf, history
            
        except Exception as e:
            logger.error(f"Erreur entraînement NCF: {e}")
            raise
    
    def evaluate_model(self, model, user_indices: np.ndarray, item_indices: np.ndarray, 
                      targets: np.ndarray, model_name: str = "model"):
        """
        Évaluer un modèle TensorFlow
        
        Args:
            model: Modèle à évaluer
            user_indices: Indices utilisateurs
            item_indices: Indices items
            targets: Targets
            model_name: Nom du modèle
        """
        try:
            # Prédictions
            predictions = model.predict([user_indices, item_indices], verbose=0)
            
            # Métriques
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            # Convertir les probabilités en prédictions binaires
            binary_predictions = (predictions.flatten() >= 0.5).astype(int)
            
            accuracy = accuracy_score(targets, binary_predictions)
            precision = precision_score(targets, binary_predictions, average='weighted')
            recall = recall_score(targets, binary_predictions, average='weighted')
            f1 = f1_score(targets, binary_predictions, average='weighted')
            auc = roc_auc_score(targets, predictions.flatten())
            
            results = {
                'model_name': model_name,
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc': float(auc),
                'predictions': predictions.flatten().tolist(),
                'binary_predictions': binary_predictions.tolist()
            }
            
            self.evaluation_results[model_name] = results
            
            logger.info(f"Évaluation {model_name}: Accuracy={accuracy:.4f}, AUC={auc:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur évaluation modèle: {e}")
            return None
    
    def plot_training_history(self, history: Dict, model_name: str = "model"):
        """Visualiser l'historique d'entraînement"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Loss
            axes[0].plot(history['loss'], label='Training Loss')
            axes[0].plot(history['val_loss'], label='Validation Loss')
            axes[0].set_title(f'{model_name} - Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True)
            
            # Accuracy
            if 'accuracy' in history:
                axes[1].plot(history['accuracy'], label='Training Accuracy')
                axes[1].plot(history['val_accuracy'], label='Validation Accuracy')
                axes[1].set_title(f'{model_name} - Accuracy')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Accuracy')
                axes[1].legend()
                axes[1].grid(True)
            elif 'auc' in history:
                axes[1].plot(history['auc'], label='Training AUC')
                axes[1].plot(history['val_auc'], label='Validation AUC')
                axes[1].set_title(f'{model_name} - AUC')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('AUC')
                axes[1].legend()
                axes[1].grid(True)
            
            plt.tight_layout()
            
            # Sauvegarder
            plt.savefig(f"data/processed/training_history_{model_name}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Historique d'entraînement sauvegardé: {model_name}")
            
        except Exception as e:
            logger.error(f"Erreur visualisation historique: {e}")
    
    def create_evaluation_report(self, results: Dict, model_name: str = "model"):
        """Créer un rapport d'évaluation"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Évaluation TensorFlow - {model_name}', fontsize=16)
            
            # 1. Métriques principales
            metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
            metrics_values = [results.get(m, 0) for m in metrics_names]
            
            axes[0, 0].bar(metrics_names, metrics_values, color='steelblue')
            axes[0, 0].set_title('Métriques Principales')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. Distribution des prédictions
            predictions = np.array(results['predictions'])
            axes[0, 1].hist(predictions, bins=50, alpha=0.7, color='coral')
            axes[0, 1].set_title('Distribution des Prédictions')
            axes[0, 1].set_xlabel('Probabilité Prédite')
            axes[0, 1].set_ylabel('Count')
            
            # 3. Scatter Plot (Predictions vs Targets)
            binary_predictions = np.array(results['binary_predictions'])
            targets_sample = np.random.choice([0, 1], size=len(binary_predictions))
            
            axes[1, 0].scatter(predictions, binary_predictions, alpha=0.5)
            axes[1, 0].set_title('Prédictions vs Targets')
            axes[1, 0].set_xlabel('Probabilité Prédite')
            axes[1, 0].set_ylabel('Prédiction Binaire')
            
            # 4. Résumé textuel
            summary_text = f"""Résumé Évaluation:
            
Accuracy: {results['accuracy']:.4f}
Precision: {results['precision']:.4f}
Recall: {results['recall']:.4f}
F1 Score: {results['f1_score']:.4f}
AUC: {results['auc']:.4f}
"""
            axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, 
                           transform=axes[1, 1].transAxes, verticalalignment='center')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # Sauvegarder
            plt.savefig(f"data/processed/evaluation_report_{model_name}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Rapport d'évaluation sauvegardé: {model_name}")
            
        except Exception as e:
            logger.error(f"Erreur création rapport évaluation: {e}")
    
    def run_complete_pipeline(self, data_path: str, model_name: str = "ncf"):
        """Exécuter le pipeline complet"""
        try:
            start_time = datetime.now()
            logger.info("Démarrage pipeline TensorFlow complet")
            
            # 1. Charger les données
            df = self.load_data(data_path)
            
            # 2. Préparer les données
            (user_indices, item_indices, targets,
             user_mapping, item_mapping,
             reverse_user_mapping, reverse_item_mapping,
             num_users, num_items) = self.prepare_ncf_data(df)
            
            # 3. Split train/test
            from sklearn.model_selection import train_test_split
            user_train, user_test, item_train, item_test, target_train, target_test = train_test_split(
                user_indices, item_indices, targets, test_size=0.2, random_state=42
            )
            
            # 4. Entraîner le modèle
            ncf_model, history = self.train_ncf_model(
                user_train, item_train, target_train,
                num_users, num_items,
                embedding_dim=64,
                hidden_units=[128, 64, 32],
                epochs=10,
                batch_size=256
            )
            
            # 5. Visualiser l'historique
            self.plot_training_history(history.history, model_name)
            
            # 6. Évaluer le modèle
            results = self.evaluate_model(ncf_model.model, user_test, item_test, target_test, model_name)
            
            # 7. Créer le rapport d'évaluation
            if results:
                self.create_evaluation_report(results, model_name)
            
            # Finaliser
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            pipeline_results = {
                'status': 'success',
                'duration_seconds': duration,
                'model_name': model_name,
                'evaluation_results': results,
                'data_stats': {
                    'total_interactions': len(df),
                    'num_users': num_users,
                    'num_items': num_items,
                    'train_samples': len(user_train),
                    'test_samples': len(user_test)
                }
            }
            
            logger.info(f"Pipeline TensorFlow terminé en {duration:.2f}s")
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Erreur pipeline TensorFlow: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

def main():
    """Fonction principale"""
    pipeline = TensorFlowTrainingPipeline()
    
    # Configuration des datasets
    datasets = [
        {
            'data_path': "data/processed/electronics_features_feature_store.csv",
            'model_name': 'ncf_electronics'
        },
        {
            'data_path': "data/processed/clothing_features_feature_store.csv",
            'model_name': 'ncf_clothing'
        }
    ]
    
    results = []
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"TensorFlow Pipeline: {dataset['model_name']}")
        print(f"{'='*60}")
        
        result = pipeline.run_complete_pipeline(
            dataset['data_path'],
            dataset['model_name']
        )
        
        results.append(result)
        
        if result['status'] == 'success':
            print(f"✅ {dataset['model_name']}: Pipeline terminé")
            
            if result['evaluation_results']:
                eval_results = result['evaluation_results']
                print(f"   Accuracy: {eval_results['accuracy']:.4f}")
                print(f"   AUC: {eval_results['auc']:.4f}")
            
            print(f"   Duration: {result['duration_seconds']:.2f}s")
        else:
            print(f"❌ {dataset['model_name']}: {result['error']}")
    
    # Rapport final
    print(f"\n{'='*60}")
    print("RAPPORT PIPELINE TENSORFLOW FINAL")
    print(f"{'='*60}")
    
    successful_pipelines = [r for r in results if r['status'] == 'success']
    
    if successful_pipelines:
        print(f"Pipelines réussis: {len(successful_pipelines)}/{len(results)}")
        
        # Moyennes des performances
        accuracies = [r['evaluation_results']['accuracy'] for r in successful_pipelines]
        aucs = [r['evaluation_results']['auc'] for r in successful_pipelines]
        
        if accuracies:
            avg_accuracy = np.mean(accuracies)
            print(f"Accuracy moyenne: {avg_accuracy:.4f}")
        
        if aucs:
            avg_auc = np.mean(aucs)
            print(f"AUC moyen: {avg_auc:.4f}")
    else:
        print("❌ Aucun pipeline réussi")
    
    return results

if __name__ == "__main__":
    main()
