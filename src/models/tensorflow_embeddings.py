"""
Embeddings TensorFlow pour utilisateurs et produits
Création et visualisation des embeddings
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import json
import os

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    """Modèle d'embeddings pour utilisateurs et produits"""
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        """
        Initialiser le modèle d'embeddings
        
        Args:
            num_users: Nombre d'utilisateurs
            num_items: Nombre d'items
            embedding_dim: Dimension des embeddings
        """
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.user_embeddings = None
        self.item_embeddings = None
        self.user_mapping = {}
        self.item_mapping = {}
        
    def build_embedding_model(self):
        """Construire le modèle d'embeddings"""
        try:
            # User embedding layer
            user_embedding_layer = layers.Embedding(
                input_dim=self.num_users,
                output_dim=self.embedding_dim,
                name='user_embeddings'
            )
            
            # Item embedding layer
            item_embedding_layer = layers.Embedding(
                input_dim=self.num_items,
                output_dim=self.embedding_dim,
                name='item_embeddings'
            )
            
            # Create model to extract embeddings
            user_input = layers.Input(shape=(1,), name='user_input')
            item_input = layers.Input(shape=(1,), name='item_input')
            
            user_emb = layers.Flatten()(user_embedding_layer(user_input))
            item_emb = layers.Flatten()(item_embedding_layer(item_input))
            
            # Dot product for recommendation
            dot_product = layers.Dot(axes=1)([user_emb, item_emb])
            
            self.embedding_model = keras.Model(
                inputs=[user_input, item_input],
                outputs=dot_product,
                name='Embedding_Model'
            )
            
            # Separate models for extracting embeddings
            self.user_embedding_extractor = keras.Model(
                inputs=user_input,
                outputs=user_emb,
                name='User_Embedding_Extractor'
            )
            
            self.item_embedding_extractor = keras.Model(
                inputs=item_input,
                outputs=item_emb,
                name='Item_Embedding_Extractor'
            )
            
            self.user_embeddings = user_embedding_layer
            self.item_embeddings = item_embedding_layer
            
            logger.info("Modèle d'embeddings construit")
            
        except Exception as e:
            logger.error(f"Erreur construction modèle embeddings: {e}")
            raise
    
    def train_embeddings(self, user_indices: np.ndarray, item_indices: np.ndarray, 
                        targets: np.ndarray, epochs: int = 10, batch_size: int = 256):
        """
        Entraîner les embeddings
        
        Args:
            user_indices: Indices utilisateurs
            item_indices: Indices items
            targets: Targets (ratings ou binary)
            epochs: Nombre d'epochs
            batch_size: Taille de batch
        """
        try:
            if self.embedding_model is None:
                self.build_embedding_model()
            
            # Compiler le modèle
            self.embedding_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            # Callbacks
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
            
            # Training
            history = self.embedding_model.fit(
                [user_indices, item_indices],
                targets,
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=1
            )
            
            logger.info(f"Entraînement embeddings terminé")
            return history
            
        except Exception as e:
            logger.error(f"Erreur entraînement embeddings: {e}")
            raise
    
    def get_user_embedding(self, user_idx: int) -> np.ndarray:
        """Obtenir l'embedding d'un utilisateur"""
        try:
            if self.user_embedding_extractor is None:
                raise ValueError("Modèle non entraîné")
            
            embedding = self.user_embedding_extractor.predict(
                np.array([user_idx]),
                verbose=0
            )
            
            return embedding[0]
            
        except Exception as e:
            logger.error(f"Erreur récupération embedding utilisateur: {e}")
            return np.zeros(self.embedding_dim)
    
    def get_item_embedding(self, item_idx: int) -> np.ndarray:
        """Obtenir l'embedding d'un item"""
        try:
            if self.item_embedding_extractor is None:
                raise ValueError("Modèle non entraîné")
            
            embedding = self.item_embedding_extractor.predict(
                np.array([item_idx]),
                verbose=0
            )
            
            return embedding[0]
            
        except Exception as e:
            logger.error(f"Erreur récupération embedding item: {e}")
            return np.zeros(self.embedding_dim)
    
    def get_all_user_embeddings(self) -> np.ndarray:
        """Obtenir tous les embeddings utilisateurs"""
        try:
            if self.user_embeddings is None:
                raise ValueError("Embeddings non entraînés")
            
            embeddings = self.user_embeddings.get_weights()[0]
            return embeddings
            
        except Exception as e:
            logger.error(f"Erreur récupération embeddings utilisateurs: {e}")
            return np.zeros((self.num_users, self.embedding_dim))
    
    def get_all_item_embeddings(self) -> np.ndarray:
        """Obtenir tous les embeddings items"""
        try:
            if self.item_embeddings is None:
                raise ValueError("Embeddings non entraînés")
            
            embeddings = self.item_embeddings.get_weights()[0]
            return embeddings
            
        except Exception as e:
            logger.error(f"Erreur récupération embeddings items: {e}")
            return np.zeros((self.num_items, self.embedding_dim))
    
    def find_similar_items(self, item_idx: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Trouver les items similaires basés sur les embeddings
        
        Args:
            item_idx: Index de l'item de référence
            top_k: Nombre d'items similaires
            
        Returns:
            Liste de (item_idx, similarité)
        """
        try:
            # Obtenir tous les embeddings items
            all_embeddings = self.get_all_item_embeddings()
            
            # Obtenir l'embedding de l'item de référence
            target_embedding = all_embeddings[item_idx]
            
            # Calculer les similarités (cosine similarity)
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity([target_embedding], all_embeddings)[0]
            
            # Créer liste de (item_idx, similarité)
            item_similarities = [(idx, similarities[idx]) for idx in range(len(similarities))]
            
            # Exclure l'item lui-même
            item_similarities = [(idx, sim) for idx, sim in item_similarities if idx != item_idx]
            
            # Trier par similarité
            item_similarities.sort(key=lambda x: x[1], reverse=True)
            
            return item_similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Erreur recherche items similaires: {e}")
            return []
    
    def find_similar_users(self, user_idx: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Trouver les utilisateurs similaires basés sur les embeddings
        
        Args:
            user_idx: Index de l'utilisateur de référence
            top_k: Nombre d'utilisateurs similaires
            
        Returns:
            Liste de (user_idx, similarité)
        """
        try:
            # Obtenir tous les embeddings utilisateurs
            all_embeddings = self.get_all_user_embeddings()
            
            # Obtenir l'embedding de l'utilisateur de référence
            target_embedding = all_embeddings[user_idx]
            
            # Calculer les similarités
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity([target_embedding], all_embeddings)[0]
            
            # Créer liste de (user_idx, similarité)
            user_similarities = [(idx, similarities[idx]) for idx in range(len(similarities))]
            
            # Exclure l'utilisateur lui-même
            user_similarities = [(idx, sim) for idx, sim in user_similarities if idx != user_idx]
            
            # Trier par similarité
            user_similarities.sort(key=lambda x: x[1], reverse=True)
            
            return user_similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Erreur recherche utilisateurs similaires: {e}")
            return []
    
    def save_embeddings(self, save_path: str):
        """Sauvegarder les embeddings"""
        try:
            os.makedirs(save_path, exist_ok=True)
            
            # Sauvegarder les embeddings
            np.save(f"{save_path}/user_embeddings.npy", self.get_all_user_embeddings())
            np.save(f"{save_path}/item_embeddings.npy", self.get_all_item_embeddings())
            
            # Sauvegarder les mappings
            mappings = {
                'user_mapping': self.user_mapping,
                'item_mapping': self.item_mapping,
                'num_users': self.num_users,
                'num_items': self.num_items,
                'embedding_dim': self.embedding_dim
            }
            
            with open(f"{save_path}/mappings.json", 'w') as f:
                json.dump(mappings, f, indent=2)
            
            logger.info(f"Embeddings sauvegardés dans: {save_path}")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde embeddings: {e}")
            raise
    
    def load_embeddings(self, load_path: str):
        """Charger les embeddings"""
        try:
            # Charger les embeddings
            user_embeddings = np.load(f"{load_path}/user_embeddings.npy")
            item_embeddings = np.load(f"{load_path}/item_embeddings.npy")
            
            # Charger les mappings
            with open(f"{load_path}/mappings.json", 'r') as f:
                mappings = json.load(f)
            
            self.user_mapping = mappings['user_mapping']
            self.item_mapping = mappings['item_mapping']
            self.num_users = mappings['num_users']
            self.num_items = mappings['num_items']
            self.embedding_dim = mappings['embedding_dim']
            
            # Reconstruire le modèle
            self.build_embedding_model()
            
            # Charger les poids
            self.user_embeddings.set_weights([user_embeddings])
            self.item_embeddings.set_weights([item_embeddings])
            
            logger.info(f"Embeddings chargés depuis: {load_path}")
            
        except Exception as e:
            logger.error(f"Erreur chargement embeddings: {e}")
            raise

def main():
    """Fonction principale de démonstration"""
    # Données simulées
    np.random.seed(42)
    
    num_users = 100
    num_items = 50
    num_interactions = 1000
    
    user_indices = np.random.randint(0, num_users, num_interactions)
    item_indices = np.random.randint(0, num_items, num_interactions)
    targets = np.random.randint(1, 6, num_interactions) / 5.0  # Normalisé 0-1
    
    print("=== DÉMONSTRATION EMBEDDINGS TENSORFLOW ===\n")
    print(f"Données: {num_interactions} interactions, {num_users} users, {num_items} items\n")
    
    # Créer le modèle d'embeddings
    embedding_model = EmbeddingModel(num_users=num_users, num_items=num_items, embedding_dim=32)
    
    # Construire le modèle
    embedding_model.build_embedding_model()
    
    # Entraîner
    print("Entraînement des embeddings...")
    history = embedding_model.train_embeddings(user_indices, item_indices, targets, epochs=5, batch_size=32)
    
    print(f"\nRésultats entraînement:")
    print(f"  Loss finale: {history.history['loss'][-1]:.4f}")
    print(f"  MAE finale: {history.history['mae'][-1]:.4f}")
    
    # Obtenir des embeddings
    print("\n=== EMBEDDINGS ===")
    user_emb = embedding_model.get_user_embedding(0)
    print(f"Embedding utilisateur 0: {user_emb[:5]}... (dimension {len(user_emb)})")
    
    item_emb = embedding_model.get_item_embedding(0)
    print(f"Embedding item 0: {item_emb[:5]}... (dimension {len(item_emb)})")
    
    # Items similaires
    print("\n=== ITEMS SIMILAIRES ===")
    similar_items = embedding_model.find_similar_items(0, top_k=5)
    print(f"Top 5 items similaires à l'item 0:")
    for item_idx, similarity in similar_items:
        print(f"  Item {item_idx}: {similarity:.4f}")
    
    # Sauvegarder
    print("\n=== SAUVEGARDE ===")
    embedding_model.save_embeddings("models/embeddings_demo")
    print("Embeddings sauvegardés dans models/embeddings_demo")

if __name__ == "__main__":
    main()
