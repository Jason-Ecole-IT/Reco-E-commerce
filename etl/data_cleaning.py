"""
Pipeline de nettoyage des données Amazon Reviews
Electronics + Clothing datasets
"""

import pandas as pd
import json
import numpy as np
from datetime import datetime
import re
from typing import Dict, List, Tuple
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AmazonReviewsCleaner:
    """Classe pour nettoyer les données Amazon Reviews"""
    
    def __init__(self):
        self.stats = {
            'total_records': 0,
            'cleaned_records': 0,
            'removed_duplicates': 0,
            'removed_invalid_ratings': 0,
            'removed_empty_reviews': 0,
            'removed_spam_reviews': 0
        }
    
    def load_data(self, file_path: str, sample_size: int = None) -> pd.DataFrame:
        """
        Charger les données depuis un fichier JSON
        
        Args:
            file_path: Chemin du fichier JSON
            sample_size: Nombre d'enregistrements à charger (None = tous)
            
        Returns:
            DataFrame pandas avec les données brutes
        """
        data = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if sample_size and i >= sample_size:
                        break
                    
                    try:
                        record = json.loads(line.strip())
                        data.append(record)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Erreur JSON ligne {i}: {e}")
                        continue
                        
            df = pd.DataFrame(data)
            self.stats['total_records'] = len(df)
            logger.info(f"Chargé {len(df)} enregistrements depuis {file_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement: {e}")
            return pd.DataFrame()
    
    def clean_reviewer_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoyer les données des reviewers
        
        Args:
            df: DataFrame brut
            
        Returns:
            DataFrame avec reviewer data nettoyée
        """
        logger.info("Nettoyage des données reviewer...")
        
        # Nettoyer reviewerName
        if 'reviewerName' in df.columns:
            # Remplacer les valeurs manquantes
            df['reviewerName'] = df['reviewerName'].fillna('Anonymous')
            
            # Nettoyer les noms (supprimer caractères spéciaux)
            df['reviewerName'] = df['reviewerName'].apply(
                lambda x: re.sub(r'[^\w\s]', '', str(x)).strip() if pd.notna(x) else 'Anonymous'
            )
            
            # Supprimer les noms trop courts ou vides
            df = df[df['reviewerName'].str.len() > 1]
        
        # Valider reviewerID
        if 'reviewerID' in df.columns:
            # Supprimer les reviewerID invalides
            df = df[df['reviewerID'].notna() & (df['reviewerID'] != '')]
            df = df[df['reviewerID'].str.match(r'^[A-Z0-9]+$')]
        
        return df
    
    def clean_product_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoyer les données des produits
        
        Args:
            df: DataFrame brut
            
        Returns:
            DataFrame avec product data nettoyée
        """
        logger.info("Nettoyage des données produit...")
        
        # Valider ASIN (Amazon Standard Identification Number)
        if 'asin' in df.columns:
            # ASIN doit être une chaîne alphanumérique de 10 caractères
            df = df[df['asin'].notna() & (df['asin'] != '')]
            df = df[df['asin'].str.match(r'^[A-Z0-9]{10}$')]
        
        return df
    
    def clean_rating_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoyer les données de notation
        
        Args:
            df: DataFrame brut
            
        Returns:
            DataFrame avec rating data nettoyée
        """
        logger.info("Nettoyage des données de notation...")
        
        if 'overall' in df.columns:
            # Supprimer les notes invalides
            valid_ratings_before = len(df)
            df = df[df['overall'].notna()]
            df = df[(df['overall'] >= 1) & (df['overall'] <= 5)]
            valid_ratings_after = len(df)
            
            self.stats['removed_invalid_ratings'] = valid_ratings_before - valid_ratings_after
            logger.info(f"Supprimé {self.stats['removed_invalid_ratings']} notes invalides")
        
        # Nettoyer les votes utiles (helpful)
        if 'helpful' in df.columns:
            # S'assurer que helpful est une liste [helpful_votes, total_votes]
            def clean_helpful(helpful_data):
                try:
                    if pd.isna(helpful_data) or helpful_data is None:
                        return [0, 0]
                    if isinstance(helpful_data, list) and len(helpful_data) >= 2:
                        return [int(helpful_data[0]), int(helpful_data[1])]
                    return [0, 0]
                except (ValueError, TypeError, IndexError):
                    return [0, 0]
            
            df['helpful'] = df['helpful'].apply(clean_helpful)
        
        return df
    
    def clean_review_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoyer le texte des reviews
        
        Args:
            df: DataFrame brut
            
        Returns:
            DataFrame avec review text nettoyé
        """
        logger.info("Nettoyage du texte des reviews...")
        
        # Nettoyer reviewText
        if 'reviewText' in df.columns:
            initial_count = len(df)
            
            # Supprimer les reviews vides
            df = df[df['reviewText'].notna()]
            df = df[df['reviewText'].str.strip() != '']
            
            # Nettoyer le texte
            def clean_text(text):
                if pd.isna(text):
                    return ""
                
                text = str(text).strip()
                
                # Supprimer les caractères non-imprimables sauf espaces
                text = re.sub(r'[^\x20-\x7E]', '', text)
                
                # Normaliser les espaces multiples
                text = re.sub(r'\s+', ' ', text)
                
                # Supprimer les reviews trop courtes (<10 caractères)
                if len(text) < 10:
                    return np.nan
                
                return text
            
            df['reviewText'] = df['reviewText'].apply(clean_text)
            
            # Supprimer les reviews avec texte nettoyé vide
            df = df[df['reviewText'].notna()]
            
            final_count = len(df)
            self.stats['removed_empty_reviews'] = initial_count - final_count
            logger.info(f"Supprimé {self.stats['removed_empty_reviews']} reviews avec texte vide/trop court")
        
        # Nettoyer summary
        if 'summary' in df.columns:
            def clean_summary(text):
                if pd.isna(text):
                    return ""
                
                text = str(text).strip()
                text = re.sub(r'[^\x20-\x7E]', '', text)
                text = re.sub(r'\s+', ' ', text)
                
                return text[:100]  # Limiter à 100 caractères
            
            df['summary'] = df['summary'].apply(clean_summary)
            df['summary'] = df['summary'].fillna('No summary')
        
        return df
    
    def detect_spam_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Détecter et supprimer les reviews spam
        
        Args:
            df: DataFrame à analyser
            
        Returns:
            DataFrame sans reviews spam
        """
        logger.info("Détection des reviews spam...")
        
        initial_count = len(df)
        
        # Critères de détection de spam (moins stricts)
        def is_spam(row):
            review_text = str(row.get('reviewText', ''))
            
            # Reviews trop courtes (<10 caractères)
            if len(review_text) < 10:
                return True
            
            # Reviews avec caractères répétitifs extrêmes (<10% de caractères uniques)
            if len(set(review_text)) < len(review_text) * 0.1:
                return True
            
            # Reviews avec beaucoup de majuscules (>90%)
            if review_text and sum(c.isupper() for c in review_text) / len(review_text) > 0.9:
                return True
            
            # Reviews avec motifs suspects évidents
            spam_patterns = [
                r'http[s]?://\S+',  # URLs
                r'www\.\S+',        # URLs
                r'click here',       # Call to action explicite
                r'buy\s+now',       # Call to action explicite
                r'free\s+trial',     # Marketing
                r'money\s+back',     # Marketing
                r'\.com'            # URLs
            ]
            
            for pattern in spam_patterns:
                if re.search(pattern, review_text, re.IGNORECASE):
                    return True
            
            return False
        
        # Appliquer la détection
        spam_mask = df.apply(is_spam, axis=1)
        spam_count = spam_mask.sum()
        
        df = df[~spam_mask]
        
        self.stats['removed_spam_reviews'] = spam_count
        logger.info(f"Détecté et supprimé {spam_count} reviews spam")
        
        return df
    
    def clean_temporal_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoyer et standardiser les données temporelles
        
        Args:
            df: DataFrame brut
            
        Returns:
            DataFrame avec temporal data nettoyée
        """
        logger.info("Nettoyage des données temporelles...")
        
        # Nettoyer unixReviewTime
        if 'unixReviewTime' in df.columns:
            df = df[df['unixReviewTime'].notna()]
            
            # Valider les timestamps (entre 1995 et 2025)
            min_timestamp = datetime(1995, 1, 1).timestamp()
            max_timestamp = datetime(2025, 12, 31).timestamp()
            
            df = df[
                (df['unixReviewTime'] >= min_timestamp) & 
                (df['unixReviewTime'] <= max_timestamp)
            ]
        
        # Nettoyer reviewTime
        if 'reviewTime' in df.columns:
            df = df[df['reviewTime'].notna()]
            # Standardiser le format
            df['reviewTime'] = pd.to_datetime(df['reviewTime'], errors='coerce')
            df = df[df['reviewTime'].notna()]
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Supprimer les doublons
        
        Args:
            df: DataFrame avec doublons potentiels
            
        Returns:
            DataFrame sans doublons
        """
        logger.info("Suppression des doublons...")
        
        initial_count = len(df)
        
        # Doublons basés sur reviewerID + asin + unixReviewTime
        df = df.drop_duplicates(subset=['reviewerID', 'asin', 'unixReviewTime'], keep='first')
        
        final_count = len(df)
        self.stats['removed_duplicates'] = initial_count - final_count
        
        logger.info(f"Supprimé {self.stats['removed_duplicates']} doublons")
        
        return df
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajouter des features dérivées
        
        Args:
            df: DataFrame nettoyé
            
        Returns:
            DataFrame avec features additionnelles
        """
        logger.info("Ajout des features dérivées...")
        
        # Longueur du review
        if 'reviewText' in df.columns:
            df['review_length'] = df['reviewText'].str.len()
            df['review_word_count'] = df['reviewText'].str.split().str.len()
        
        # Ratio helpful
        if 'helpful' in df.columns:
            df['helpful_votes'] = df['helpful'].apply(lambda x: x[0] if isinstance(x, list) and len(x) >= 2 else 0)
            df['total_votes'] = df['helpful'].apply(lambda x: x[1] if isinstance(x, list) and len(x) >= 2 else 0)
            df['helpful_ratio'] = df.apply(
                lambda row: row['helpful_votes'] / row['total_votes'] if row['total_votes'] > 0 else 0, 
                axis=1
            )
        
        # Features temporelles
        if 'unixReviewTime' in df.columns:
            df['review_date'] = pd.to_datetime(df['unixReviewTime'], unit='s')
            df['review_year'] = df['review_date'].dt.year
            df['review_month'] = df['review_date'].dt.month
            df['review_dayofweek'] = df['review_date'].dt.dayofweek
        
        # Catégorie de rating
        if 'overall' in df.columns:
            def rating_category(rating):
                if rating <= 2:
                    return 'negative'
                elif rating == 3:
                    return 'neutral'
                else:
                    return 'positive'
            
            df['rating_category'] = df['overall'].apply(rating_category)
        
        return df
    
    def clean_dataset(self, file_path: str, output_path: str = None, sample_size: int = None) -> pd.DataFrame:
        """
        Pipeline complet de nettoyage
        
        Args:
            file_path: Chemin du fichier d'entrée
            output_path: Chemin du fichier de sortie (optionnel)
            sample_size: Taille de l'échantillon (optionnel)
            
        Returns:
            DataFrame nettoyé
        """
        logger.info(f"Début du nettoyage du dataset: {file_path}")
        
        # Charger les données
        df = self.load_data(file_path, sample_size)
        
        if df.empty:
            logger.error("Impossible de charger les données")
            return df
        
        # Pipeline de nettoyage
        df = self.clean_reviewer_data(df)
        df = self.clean_product_data(df)
        df = self.clean_rating_data(df)
        df = self.clean_review_text(df)
        df = self.detect_spam_reviews(df)
        df = self.clean_temporal_data(df)
        df = self.remove_duplicates(df)
        
        # Features dérivées
        df = self.add_derived_features(df)
        
        # Statistiques finales
        self.stats['cleaned_records'] = len(df)
        
        logger.info(f"Nettoyage terminé: {self.stats['cleaned_records']}/{self.stats['total_records']} enregistrements conservés")
        logger.info(f"Taux de conservation: {self.stats['cleaned_records']/self.stats['total_records']:.2%}")
        
        # Sauvegarder si chemin de sortie spécifié
        if output_path:
            df.to_json(output_path, orient='records', lines=True)
            logger.info(f"Données nettoyées sauvegardées dans: {output_path}")
        
        return df
    
    def get_cleaning_stats(self) -> Dict:
        """Retourner les statistiques de nettoyage"""
        return self.stats

def main():
    """Fonction principale de test"""
    cleaner = AmazonReviewsCleaner()
    
    # Nettoyer les données Electronics
    electronics_input = "data/raw/amazon_reviews_electronics_url.json"
    electronics_output = "data/processed/amazon_reviews_electronics_clean.json"
    
    print("=== Nettoyage des données Electronics ===")
    df_electronics = cleaner.clean_dataset(
        electronics_input, 
        electronics_output,
        sample_size=100000  # Limiter pour le test
    )
    
    print("\nStatistiques de nettoyage Electronics:")
    stats = cleaner.get_cleaning_stats()
    for key, value in stats.items():
        print(f"  {key}: {value:,}")
    
    if not df_electronics.empty:
        print(f"\nAperçu des données nettoyées:")
        print(df_electronics[['reviewerID', 'asin', 'overall', 'review_length', 'rating_category']].head())
    
    # Nettoyer les données Clothing
    cleaner = AmazonReviewsCleaner()  # Reset stats
    
    clothing_input = "data/raw/amazon_reviews_clothing_url.json"
    clothing_output = "data/processed/amazon_reviews_clothing_clean.json"
    
    print("\n=== Nettoyage des données Clothing ===")
    df_clothing = cleaner.clean_dataset(
        clothing_input,
        clothing_output,
        sample_size=50000  # Limiter pour le test
    )
    
    print("\nStatistiques de nettoyage Clothing:")
    stats = cleaner.get_cleaning_stats()
    for key, value in stats.items():
        print(f"  {key}: {value:,}")

if __name__ == "__main__":
    main()
