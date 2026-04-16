"""
Neural Collaborative Filtering (NCF) avec TensorFlow
Implementation de NCF pour recommandations e-commerce
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import os
import json

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NCFModel:
    """Neural Collaborative Filtering Model"""
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        """
        Initialiser le modèle NCF
        
        Args:
            num_users: Nombre d'utilisateurs uniques
            num_items: Nombre d'items uniques
            embedding_dim: Dimension des embeddings
        """
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.model = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        
    def build_model(self, hidden_units: List[int] = [128, 64, 32], dropout_rate: float = 0.2):
        """
        Construire l'architecture NCF
        
        Args:
            hidden_units: Liste des unités cachées
            dropout_rate: Taux de dropout
        """
        try:
            # Input layers
            user_input = layers.Input(shape=(1,), name='user_id')
            item_input = layers.Input(shape=(1,), name='item_id')
            
            # Embedding layers
            user_embedding = layers.Embedding(
                input_dim=self.num_users,
                output_dim=self.embedding_dim,
                name='user_embedding'
            )(user_input)
            
            item_embedding = layers.Embedding(
                input_dim=self.num_items,
                output_dim=self.embedding_dim,
                name='item_embedding'
            )(item_input)
            
            # Flatten embeddings
            user_vec = layers.Flatten()(user_embedding)
            item_vec = layers.Flatten()(item_embedding)
            
            # Concatenate user and item embeddings
            concat = layers.Concatenate()([user_vec, item_vec])
            
            # MLP layers
            dense = concat
            for units in hidden_units:
                dense = layers.Dense(units, activation='relu')(dense)
                dense = layers.Dropout(dropout_rate)(dense)
            
            # Output layer (rating prediction)
            output = layers.Dense(1, activation='sigmoid')(dense)
            
            # Create model
            self.model = models.Model(
                inputs=[user_input, item_input],
                outputs=output,
                name='NCF_Model'
            )
            
            # Compile model
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.AUC()]
            )
            
            logger.info(f"Modèle NCF construit: {self.model.summary()}")
            
        except Exception as e:
            logger.error(f"Erreur construction modèle: {e}")
            raise
    
    def create_mappings(self, user_ids: List[str], item_ids: List[str]):
        """
        Créer les mappings user/item vers indices
        
        Args:
            user_ids: Liste des IDs utilisateurs
            item_ids: Liste des IDs items
        """
        try:
            # User mappings
            unique_users = sorted(list(set(user_ids)))
            self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
            self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
            
            # Item mappings
            unique_items = sorted(list(set(item_ids)))
            self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}
            self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}
            
            # Update dimensions
            self.num_users = len(unique_users)
            self.num_items = len(unique_items)
            
            logger.info(f"Mappings créés: {len(unique_users)} users, {len(unique_items)} items")
            
        except Exception as e:
            logger.error(f"Erreur création mappings: {e}")
            raise
    
    def prepare_data(self, df: pd.DataFrame, user_col: str = 'reviewerID', item_col: str = 'asin', rating_col: str = 'overall'):
        """
        Préparer les données pour l'entraînement
        
        Args:
            df: DataFrame avec les interactions
            user_col: Colonne utilisateur
            item_col: Colonne item
            rating_col: Colonne rating
        """
        try:
            # Créer les mappings si pas encore fait
            if not self.user_mapping or not self.item_mapping:
                self.create_mappings(df[user_col].tolist(), df[item_col].tolist())
            
            # Convertir IDs en indices
            df['user_idx'] = df[user_col].map(self.user_mapping)
            df['item_idx'] = df[item_col].map(self.item_mapping)
            
            # Créer target binaire (rating >= 4 = 1, sinon 0)
            df['target'] = (df[rating_col] >= 4).astype(int)
            
            # Supprimer les valeurs manquantes
            df = df.dropna(subset=['user_idx', 'item_idx'])
            
            # Convertir en numpy arrays
            user_indices = df['user_idx'].values
            item_indices = df['item_idx'].values
            targets = df['target'].values
            
            logger.info(f"Données préparées: {len(df)} interactions")
            
            return user_indices, item_indices, targets
            
        except Exception as e:
            logger.error(f"Erreur préparation données: {e}")
            raise
    
    def train(self, user_indices: np.ndarray, item_indices: np.ndarray, targets: np.ndarray,
              validation_split: float = 0.2, epochs: int = 10, batch_size: int = 256):
        """
        Entraîner le modèle NCF
        
        Args:
            user_indices: Indices utilisateurs
            item_indices: Indices items
            targets: Targets binaires
            validation_split: Ratio validation
            epochs: Nombre d'epochs
            batch_size: Taille de batch
        """
        try:
            if self.model is None:
                raise ValueError("Modèle non construit. Appelez build_model() d'abord.")
            
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
            
            # Training
            history = self.model.fit(
                [user_indices, item_indices],
                targets,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            logger.info(f"Entraînement terminé: {len(history.history['loss'])} epochs")
            
            return history
            
        except Exception as e:
            logger.error(f"Erreur entraînement: {e}")
            raise
    
    def predict(self, user_id: str, item_id: str) -> float:
        """
        Prédire la probabilité d'interaction
        
        Args:
            user_id: ID utilisateur
            item_id: ID item
            
        Returns:
            Probabilité prédite
        """
        try:
            if user_id not in self.user_mapping or item_id not in self.item_mapping:
                return 0.5  # Valeur par défaut
            
            user_idx = self.user_mapping[user_id]
            item_idx = self.item_mapping[item_id]
            
            # Prédiction
            prediction = self.model.predict([
                np.array([user_idx]),
                np.array([item_idx])
            ], verbose=0)
            
            return float(prediction[0][0])
            
        except Exception as e:
            logger.error(f"Erreur prédiction: {e}")
            return 0.5
    
    def recommend(self, user_id: str, top_k: int = 10, exclude_seen: List[str] = None) -> List[Tuple[str, float]]:
        """
        Générer des recommandations pour un utilisateur
        
        Args:
            user_id: ID utilisateur
            top_k: Nombre de recommandations
            exclude_seen: Items à exclure
            
        Returns:
            Liste de (item_id, score)
        """
        try:
            if user_id not in self.user_mapping:
                # Retourner les items les plus populaires
                return self._get_popular_items(top_k)
            
            user_idx = self.user_mapping[user_id]
            
            # Prédire pour tous les items
            item_indices = np.arange(self.num_items)
            user_indices_array = np.full(self.num_items, user_idx)
            
            # Batch prediction
            predictions = self.model.predict(
                [user_indices_array, item_indices],
                batch_size=1024,
                verbose=0
            )
            
            # Créer liste de (item_id, score)
            item_scores = [
                (self.reverse_item_mapping[idx], float(predictions[idx][0]))
                for idx in range(self.num_items)
            ]
            
            # Exclure les items déjà vus
            if exclude_seen:
                item_scores = [(item, score) for item, score in item_scores if item not in exclude_seen]
            
            # Trier par score et prendre top_k
            item_scores.sort(key=lambda x: x[1], reverse=True)
            
            return item_scores[:top_k]
            
        except Exception as e:
            logger.error(f"Erreur recommandations: {e}")
            return self._get_popular_items(top_k)
    
    def _get_popular_items(self, top_k: int) -> List[Tuple[str, float]]:
        """Retourner les items les plus populaires (fallback)"""
        # Pour l'instant, retourner des items aléatoires avec scores constants
        items = list(self.reverse_item_mapping.values())
        return [(item, 0.5) for item in items[:top_k]]
    
    def save_model(self, model_path: str):
        """Sauvegarder le modèle et les mappings"""
        try:
            # Créer le répertoire
            os.makedirs(model_path, exist_ok=True)
            
            # Sauvegarder le modèle
            self.model.save(f"{model_path}/ncf_model.h5")
            
            # Sauvegarder les mappings
            mappings = {
                'user_mapping': self.user_mapping,
                'item_mapping': self.item_mapping,
                'reverse_user_mapping': self.reverse_user_mapping,
                'reverse_item_mapping': self.reverse_item_mapping,
                'num_users': self.num_users,
                'num_items': self.num_items,
                'embedding_dim': self.embedding_dim
            }
            
            with open(f"{model_path}/mappings.json", 'w') as f:
                json.dump(mappings, f, indent=2)
            
            logger.info(f"Modèle sauvegardé dans: {model_path}")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde modèle: {e}")
            raise
    
    def load_model(self, model_path: str):
        """Charger le modèle et les mappings"""
        try:
            # Charger les mappings
            with open(f"{model_path}/mappings.json", 'r') as f:
                mappings = json.load(f)
            
            self.user_mapping = mappings['user_mapping']
            self.item_mapping = mappings['item_mapping']
            self.reverse_user_mapping = {int(k): v for k, v in mappings['reverse_user_mapping'].items()}
            self.reverse_item_mapping = {int(k): v for k, v in mappings['reverse_item_mapping'].items()}
            self.num_users = mappings['num_users']
            self.num_items = mappings['num_items']
            self.embedding_dim = mappings['embedding_dim']
            
            # Reconstruire le modèle
            self.build_model()
            
            # Charger les poids
            self.model.load_weights(f"{model_path}/ncf_model.h5")
            
            logger.info(f"Modèle chargé depuis: {model_path}")
            
        except Exception as e:
            logger.error(f"Erreur chargement modèle: {e}")
            raise

def main():
    """Fonction principale de démonstration"""
    # Créer des données simulées
    np.random.seed(42)
    
    num_users = 100
    num_items = 50
    num_interactions = 1000
    
    # Générer des interactions simulées
    user_ids = [f"user_{i}" for i in np.random.randint(0, num_users, num_interactions)]
    item_ids = [f"item_{i}" for i in np.random.randint(0, num_items, num_interactions)]
    ratings = np.random.randint(1, 6, num_interactions)
    
    df = pd.DataFrame({
        'reviewerID': user_ids,
        'asin': item_ids,
        'overall': ratings
    })
    
    print("=== DÉMONSTRATION NCF TENSORFLOW ===\n")
    print(f"Données simulées: {num_interactions} interactions, {num_users} users, {num_items} items\n")
    
    # Créer le modèle
    ncf = NCFModel(num_users=num_users, num_items=num_items, embedding_dim=32)
    
    # Préparer les données
    user_indices, item_indices, targets = ncf.prepare_data(df)
    
    # Construire le modèle
    ncf.build_model(hidden_units=[64, 32], dropout_rate=0.2)
    
    # Entraîner
    print("Entraînement du modèle...")
    history = ncf.train(user_indices, item_indices, targets, epochs=5, batch_size=32)
    
    print(f"\nRésultats entraînement:")
    print(f"  Loss finale: {history.history['loss'][-1]:.4f}")
    print(f"  Accuracy finale: {history.history['accuracy'][-1]:.4f}")
    print(f"  AUC finale: {history.history['auc'][-1]:.4f}")
    
    # Prédiction
    print("\n=== PRÉDICTIONS ===")
    test_user = user_ids[0]
    test_item = item_ids[0]
    prediction = ncf.predict(test_user, test_item)
    print(f"Prédiction pour {test_user} / {test_item}: {prediction:.4f}")
    
    # Recommandations
    print("\n=== RECOMMANDATIONS ===")
    recommendations = ncf.recommend(test_user, top_k=5)
    print(f"Top 5 recommandations pour {test_user}:")
    for item, score in recommendations:
        print(f"  {item}: {score:.4f}")
    
    # Sauvegarder le modèle
    print("\n=== SAUVEGARDE ===")
    ncf.save_model("models/ncf_demo")
    print("Modèle sauvegardé dans models/ncf_demo")

if __name__ == "__main__":
    main()
