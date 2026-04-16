"""
Feature Engineering avancé pour le moteur de recommandation
Aggregations temporelles, window functions, feature selection
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import re

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature Engineer avancé pour recommandation systems"""
    
    def __init__(self):
        """Initialiser le feature engineer"""
        self.feature_stats = {
            'start_time': None,
            'end_time': None,
            'features_created': [],
            'features_selected': [],
            'errors': []
        }
    
    def load_transformed_data(self, csv_path: str):
        """Charger les données transformées du pipeline ETL"""
        try:
            logger.info(f"Chargement des données transformées: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Conversion des dates
            if 'review_timestamp' in df.columns:
                df['review_timestamp'] = pd.to_datetime(df['review_timestamp'])
            
            logger.info(f"Chargé {len(df):,} enregistrements")
            return df
            
        except Exception as e:
            error_msg = f"Erreur chargement données: {e}"
            logger.error(error_msg)
            self.feature_stats['errors'].append(error_msg)
            raise
    
    def create_temporal_features(self, df):
        """Créer des features temporelles avancées"""
        try:
            logger.info("Création des features temporelles")
            
            # S'assurer que review_timestamp est en datetime
            if 'review_timestamp' not in df.columns:
                logger.warning("Colonne review_timestamp non trouvée")
                return df
            
            df_temp = df.copy()
            
            # 1. Features temporelles de base
            df_temp['review_year'] = df_temp['review_timestamp'].dt.year
            df_temp['review_month'] = df_temp['review_timestamp'].dt.month
            df_temp['review_day'] = df_temp['review_timestamp'].dt.day
            df_temp['review_weekday'] = df_temp['review_timestamp'].dt.weekday
            df_temp['review_hour'] = df_temp['review_timestamp'].dt.hour
            df_temp['review_dayofyear'] = df_temp['review_timestamp'].dt.dayofyear
            
            # 2. Features de saisonnalité
            df_temp['is_weekend'] = (df_temp['review_weekday'] >= 5).astype(int)
            df_temp['is_holiday'] = self._is_holiday_season(df_temp['review_month'])
            
            # 3. Features cycliques (sin/cos pour capturer la saisonnalité)
            df_temp['month_sin'] = np.sin(2 * np.pi * df_temp['review_month'] / 12)
            df_temp['month_cos'] = np.cos(2 * np.pi * df_temp['review_month'] / 12)
            df_temp['weekday_sin'] = np.sin(2 * np.pi * df_temp['review_weekday'] / 7)
            df_temp['weekday_cos'] = np.cos(2 * np.pi * df_temp['review_weekday'] / 7)
            
            # 4. Features de récence
            current_date = df_temp['review_timestamp'].max()
            df_temp['days_since_last_review'] = (current_date - df_temp['review_timestamp']).dt.days
            df_temp['weeks_since_last_review'] = df_temp['days_since_last_review'] // 7
            
            # 5. Features de périodicité utilisateur
            user_time_features = df_temp.groupby('reviewerID').agg({
                'review_timestamp': ['min', 'max', 'count']
            }).round(2)
            
            user_time_features.columns = ['user_first_review', 'user_last_review', 'user_review_count']
            
            # Calculer la durée d'activité utilisateur
            user_time_features['user_activity_days'] = (
                user_time_features['user_last_review'] - user_time_features['user_first_review']
            ).dt.days
            
            # Calculer la fréquence de review (reviews par jour)
            user_time_features['user_review_frequency'] = (
                user_time_features['user_review_count'] / 
                (user_time_features['user_activity_days'] + 1)
            )
            
            # Joindre les features temporelles utilisateur
            df_temp = df_temp.merge(
                user_time_features[['user_activity_days', 'user_review_frequency']], 
                left_on='reviewerID', 
                right_index=True, 
                how='left'
            )
            
            # 6. Features de périodicité produit
            product_time_features = df_temp.groupby('asin').agg({
                'review_timestamp': ['min', 'max', 'count']
            }).round(2)
            
            product_time_features.columns = ['product_first_review', 'product_last_review', 'product_review_count']
            
            # Calculer la durée de vie produit
            product_time_features['product_lifetime_days'] = (
                product_time_features['product_last_review'] - product_time_features['product_first_review']
            ).dt.days
            
            # Joindre les features temporelles produit
            df_temp = df_temp.merge(
                product_time_features[['product_lifetime_days']], 
                left_on='asin', 
                right_index=True, 
                how='left'
            )
            
            self.feature_stats['features_created'].extend([
                'review_year', 'review_month', 'review_day', 'review_weekday', 'review_hour',
                'review_dayofyear', 'is_weekend', 'is_holiday',
                'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos',
                'days_since_last_review', 'weeks_since_last_review',
                'user_activity_days', 'user_review_frequency', 'product_lifetime_days'
            ])
            
            logger.info("Features temporelles créées avec succès")
            return df_temp
            
        except Exception as e:
            error_msg = f"Erreur création features temporelles: {e}"
            logger.error(error_msg)
            self.feature_stats['errors'].append(error_msg)
            raise
    
    def create_interaction_features(self, df):
        """Créer des features d'interaction utilisateur-produit"""
        try:
            logger.info("Création des features d'interaction")
            
            df_inter = df.copy()
            
            # 1. Features de popularité croisée
            user_popularity = df_inter.groupby('reviewerID')['user_review_count'].first()
            product_popularity = df_inter.groupby('asin')['product_review_count'].first()
            
            df_inter['user_popularity'] = df_inter['reviewerID'].map(user_popularity)
            df_inter['product_popularity'] = df_inter['asin'].map(product_popularity)
            
            # 2. Features de diversité utilisateur
            user_diversity = df_inter.groupby('reviewerID').agg({
                'asin': 'nunique',
                'overall': ['mean', 'std']
            }).round(2)
            
            user_diversity.columns = ['user_product_diversity', 'user_avg_rating', 'user_rating_std']
            df_inter = df_inter.merge(
                user_diversity, 
                left_on='reviewerID', 
                right_index=True, 
                how='left'
            )
            
            # 3. Features de consensus produit
            product_consensus = df_inter.groupby('asin').agg({
                'reviewerID': 'nunique',
                'overall': ['mean', 'std']
            }).round(2)
            
            product_consensus.columns = ['product_user_diversity', 'product_avg_rating', 'product_rating_std']
            df_inter = df_inter.merge(
                product_consensus, 
                left_on='asin', 
                right_index=True, 
                how='left'
            )
            
            # 4. Features de confiance
            # Vérifier si les colonnes existent
            if 'user_rating_std' in df_inter.columns and 'product_rating_std' in df_inter.columns:
                df_inter['rating_confidence'] = np.where(
                    (df_inter['user_rating_std'] < 1.5) & (df_inter['product_rating_std'] < 1.5),
                    1.0,  # haute confiance
                    np.where(
                        (df_inter['user_rating_std'] < 2.0) & (df_inter['product_rating_std'] < 2.0),
                        0.7,  # moyenne confiance
                        0.3   # basse confiance
                    )
                )
            else:
                df_inter['rating_confidence'] = 0.7  # valeur par défaut
            
            # 5. Features d'engagement avancées
            df_inter['engagement_intensity'] = (
                df_inter['helpfulness_ratio'] * 
                df_inter['word_count'] / 100 *  # normaliser par 100 mots
                df_inter['rating_confidence']
            )
            
            # 6. Features de rareté
            user_rarity = df_inter.groupby('reviewerID').size()
            product_rarity = df_inter.groupby('asin').size()
            
            # Calculer les percentiles de rareté
            user_rarity_percentile = user_rarity.rank(pct=True)
            product_rarity_percentile = product_rarity.rank(pct=True)
            
            df_inter['user_rarity_score'] = df_inter['reviewerID'].map(user_rarity_percentile)
            df_inter['product_rarity_score'] = df_inter['asin'].map(product_rarity_percentile)
            
            # 7. Features de matching
            df_inter['popularity_match'] = (
                df_inter['user_popularity'] * df_inter['product_popularity']
            )
            
            df_inter['diversity_match'] = (
                df_inter['user_product_diversity'] * df_inter['product_user_diversity']
            )
            
            self.feature_stats['features_created'].extend([
                'user_popularity', 'product_popularity', 'user_product_diversity', 'user_avg_rating',
                'user_rating_std', 'product_user_diversity', 'product_avg_rating', 
                'product_rating_std', 'rating_confidence', 'engagement_intensity',
                'user_rarity_score', 'product_rarity_score', 'popularity_match', 'diversity_match'
            ])
            
            logger.info("Features d'interaction créées avec succès")
            return df_inter
            
        except Exception as e:
            error_msg = f"Erreur création features interaction: {e}"
            logger.error(error_msg)
            self.feature_stats['errors'].append(error_msg)
            raise
    
    def create_content_features(self, df):
        """Créer des features basées sur le contenu des reviews"""
        try:
            logger.info("Création des features de contenu")
            
            df_content = df.copy()
            
            # 1. Features de texte avancées
            df_content['text_complexity'] = df_content['reviewText_clean'].apply(self._calculate_text_complexity)
            df_content['sentiment_intensity'] = df_content['reviewText_clean'].apply(self._calculate_sentiment_intensity)
            
            # 2. Features de qualité du contenu
            df_content['review_completeness'] = (
                (df_content['word_count'] > 10).astype(int) * 0.3 +
                (df_content['char_count'] > 50).astype(int) * 0.3 +
                (df_content['helpfulness_ratio'] > 0.5).astype(int) * 0.4
            )
            
            # 3. Features de catégories croisées
            df_content['rating_length_interaction'] = (
                df_content['overall'] * np.log1p(df_content['word_count'])
            )
            
            df_content['sentiment_length_correlation'] = (
                df_content['sentiment_score'] * df_content['word_count'] / 100
            )
            
            # 4. Features de temporalité du contenu
            df_content['content_freshness'] = np.where(
                df_content['days_since_last_review'] < 30, 1.0,
                np.where(df_content['days_since_last_review'] < 90, 0.7, 0.3)
            )
            
            # 5. Features de cohérence
            df_content['rating_sentiment_consistency'] = np.where(
                (df_content['overall'] >= 4) & (df_content['sentiment_score'] > 0), 1.0,
                np.where(
                    (df_content['overall'] <= 2) & (df_content['sentiment_score'] < 0), 1.0,
                    np.where(df_content['overall'] == 3, 0.5, 0.0)
                )
            )
            
            self.feature_stats['features_created'].extend([
                'text_complexity', 'sentiment_intensity', 'review_completeness',
                'rating_length_interaction', 'sentiment_length_correlation',
                'content_freshness', 'rating_sentiment_consistency'
            ])
            
            logger.info("Features de contenu créées avec succès")
            return df_content
            
        except Exception as e:
            error_msg = f"Erreur création features contenu: {e}"
            logger.error(error_msg)
            self.feature_stats['errors'].append(error_msg)
            raise
    
    def create_aggregate_features(self, df):
        """Créer des features agrégées avec window functions"""
        try:
            logger.info("Création des features agrégées")
            
            df_agg = df.copy()
            
            # 1. Features utilisateur agrégées (simplifiées)
            # Trier par utilisateur et temps
            df_agg = df_agg.sort_values(['reviewerID', 'review_timestamp'])
            
            # Calculer les rolling features manuellement
            user_features = []
            for user_id in df_agg['reviewerID'].unique():
                user_data = df_agg[df_agg['reviewerID'] == user_id].copy()
                user_data = user_data.sort_values('review_timestamp')
                
                # Rolling averages
                user_data['user_rolling_avg_rating'] = user_data['overall'].rolling(window=5, min_periods=1).mean()
                user_data['user_rolling_std_rating'] = user_data['overall'].rolling(window=5, min_periods=1).std()
                user_data['user_rolling_avg_length'] = user_data['word_count'].rolling(window=5, min_periods=1).mean()
                
                # Trend du sentiment
                user_data['user_rolling_sentiment_trend'] = user_data['sentiment_score'].rolling(window=5, min_periods=1).apply(
                    lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0
                )
                
                user_features.append(user_data)
            
            df_agg = pd.concat(user_features, ignore_index=True)
            
            # 2. Features produit agrégées (simplifiées)
            product_features = []
            for product_id in df_agg['asin'].unique():
                product_data = df_agg[df_agg['asin'] == product_id].copy()
                product_data = product_data.sort_values('review_timestamp')
                
                # Rolling averages pour produits
                product_data['product_rolling_avg_rating'] = product_data['overall'].rolling(window=10, min_periods=1).mean()
                product_data['product_rolling_rating_trend'] = product_data['overall'].rolling(window=10, min_periods=1).apply(
                    lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0
                )
                product_data['product_rolling_popularity_trend'] = product_data['user_review_count'].rolling(window=10, min_periods=1).mean()
                
                product_features.append(product_data)
            
            df_agg = pd.concat(product_features, ignore_index=True)
            
            # 3. Features globales agrégées
            global_stats = {
                'global_avg_rating': df_agg['overall'].mean(),
                'global_rating_std': df_agg['overall'].std(),
                'global_avg_length': df_agg['word_count'].mean(),
                'global_avg_sentiment': df_agg['sentiment_score'].mean()
            }
            
            # Normaliser par rapport aux globales
            df_agg['rating_vs_global'] = df_agg['overall'] - global_stats['global_avg_rating']
            df_agg['length_vs_global'] = df_agg['word_count'] - global_stats['global_avg_length']
            df_agg['sentiment_vs_global'] = df_agg['sentiment_score'] - global_stats['global_avg_sentiment']
            
            # 4. Features de percentiles
            df_agg['rating_percentile'] = df_agg['overall'].rank(pct=True)
            df_agg['length_percentile'] = df_agg['word_count'].rank(pct=True)
            df_agg['helpfulness_percentile'] = df_agg['helpfulness_ratio'].rank(pct=True)
            
            self.feature_stats['features_created'].extend([
                'user_rolling_avg_rating', 'user_rolling_std_rating', 'user_rolling_avg_length',
                'user_rolling_sentiment_trend', 'product_rolling_avg_rating',
                'product_rolling_rating_trend', 'product_rolling_popularity_trend',
                'rating_vs_global', 'length_vs_global', 'sentiment_vs_global',
                'rating_percentile', 'length_percentile', 'helpfulness_percentile'
            ])
            
            logger.info("Features agrégées créées avec succès")
            return df_agg
            
        except Exception as e:
            error_msg = f"Erreur création features agrégées: {e}"
            logger.error(error_msg)
            self.feature_stats['errors'].append(error_msg)
            raise
    
    def select_features(self, df, target_col='overall', method='variance', k=20):
        """Sélectionner les features les plus pertinentes"""
        try:
            logger.info(f"Sélection de features avec méthode: {method}")
            
            # Identifier les colonnes numériques
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Exclure les colonnes ID et cibles
            exclude_cols = ['reviewerID', 'asin', 'review_timestamp', 'review_date_only', target_col]
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            if not feature_cols:
                logger.warning("Aucune feature numérique à sélectionner")
                return df, []
            
            X = df[feature_cols].fillna(0)
            y = df[target_col].fillna(df[target_col].mean())
            
            selected_features = []
            
            if method == 'variance':
                # Sélection par variance threshold
                selector = VarianceThreshold(threshold=0.01)
                X_selected = selector.fit_transform(X)
                selected_features = [feature_cols[i] for i, selected in enumerate(selector.get_support())]
                
            elif method == 'k_best':
                # Sélection des K meilleures features
                selector = SelectKBest(score_func=f_classif, k=min(k, len(feature_cols)))
                X_selected = selector.fit_transform(X, y)
                selected_features = [feature_cols[i] for i, selected in enumerate(selector.get_support())]
                
            elif method == 'correlation':
                # Sélection par corrélation avec la cible
                correlations = df[feature_cols + [target_col]].corr()[target_col].abs()
                correlations = correlations.drop(target_col).sort_values(ascending=False)
                selected_features = correlations.head(k).index.tolist()
            
            self.feature_stats['features_selected'] = selected_features
            
            logger.info(f"Features sélectionnées: {len(selected_features)}")
            logger.info(f"Features: {selected_features[:10]}...")  # Afficher les 10 premières
            
            return df, selected_features
            
        except Exception as e:
            error_msg = f"Erreur sélection features: {e}"
            logger.error(error_msg)
            self.feature_stats['errors'].append(error_msg)
            return df, []
    
    def scale_features(self, df, feature_cols, method='standard'):
        """Normaliser/standardiser les features"""
        try:
            logger.info(f"Scaling des features avec méthode: {method}")
            
            if not feature_cols:
                logger.warning("Aucune feature à scaler")
                return df
            
            X = df[feature_cols].fillna(0)
            
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Méthode de scaling non reconnue: {method}")
            
            X_scaled = scaler.fit_transform(X)
            
            # Créer les colonnes scaled
            scaled_cols = [f"{col}_scaled" for col in feature_cols]
            df_scaled = df.copy()
            
            for i, col in enumerate(feature_cols):
                df_scaled[scaled_cols[i]] = X_scaled[:, i]
            
            logger.info(f"Features scalées: {len(scaled_cols)}")
            return df_scaled, scaler
            
        except Exception as e:
            error_msg = f"Erreur scaling features: {e}"
            logger.error(error_msg)
            self.feature_stats['errors'].append(error_msg)
            return df, None
    
    def create_feature_store(self, df, selected_features, output_path):
        """Créer un feature store pour le modèle de recommandation"""
        try:
            logger.info("Création du feature store")
            
            # Sélectionner les colonnes essentielles
            essential_cols = ['reviewerID', 'asin', 'overall'] + selected_features
            
            feature_store = df[essential_cols].copy()
            
            # Ajouter des métadonnées
            feature_store['feature_version'] = '1.0'
            feature_store['created_timestamp'] = datetime.now().isoformat()
            feature_store['feature_count'] = len(selected_features)
            
            # Créer des index pour optimisation
            feature_store.set_index(['reviewerID', 'asin'], inplace=True)
            
            # Sauvegarder en différents formats
            # CSV pour compatibilité
            csv_path = f"{output_path}_feature_store.csv"
            feature_store.to_csv(csv_path)
            
            # JSON pour structure
            json_path = f"{output_path}_feature_store.json"
            feature_store.reset_index().to_json(json_path, orient='records', date_format='iso')
            
            # Parquet pour performance
            parquet_path = f"{output_path}_feature_store.parquet"
            feature_store.reset_index().to_parquet(parquet_path, index=False)
            
            logger.info(f"Feature store créé: CSV ({csv_path}), JSON ({json_path}), Parquet ({parquet_path})")
            
            return {
                'csv_path': csv_path,
                'json_path': json_path,
                'parquet_path': parquet_path,
                'feature_count': len(selected_features),
                'record_count': len(feature_store)
            }
            
        except Exception as e:
            error_msg = f"Erreur création feature store: {e}"
            logger.error(error_msg)
            self.feature_stats['errors'].append(error_msg)
            raise
    
    def run_feature_engineering(self, input_csv_path: str, output_base_path: str):
        """Exécuter le pipeline complet de feature engineering"""
        try:
            self.feature_stats['start_time'] = datetime.now()
            logger.info("Démarrage du pipeline de feature engineering")
            
            # Étape 1: Charger les données
            df = self.load_transformed_data(input_csv_path)
            
            # Étape 2: Créer les features temporelles
            df_temp = self.create_temporal_features(df)
            
            # Étape 3: Créer les features d'interaction
            df_inter = self.create_interaction_features(df_temp)
            
            # Étape 4: Créer les features de contenu
            df_content = self.create_content_features(df_inter)
            
            # Étape 5: Créer les features agrégées
            df_final = self.create_aggregate_features(df_content)
            
            # Étape 6: Sélectionner les features
            df_selected, selected_features = self.select_features(
                df_final, target_col='overall', method='k_best', k=25
            )
            
            # Étape 7: Scaler les features
            df_scaled, scaler = self.scale_features(
                df_selected, selected_features, method='standard'
            )
            
            # Étape 8: Créer le feature store
            feature_store_results = self.create_feature_store(
                df_scaled, selected_features, output_base_path
            )
            
            # Finaliser les statistiques
            self.feature_stats['end_time'] = datetime.now()
            self.feature_stats['records_processed'] = len(df_final)
            
            duration = (self.feature_stats['end_time'] - 
                       self.feature_stats['start_time']).total_seconds()
            
            logger.info(f"Feature engineering terminé en {duration:.2f}s")
            
            return {
                'status': 'success',
                'duration_seconds': duration,
                'records_processed': len(df_final),
                'features_created': len(self.feature_stats['features_created']),
                'features_selected': len(selected_features),
                'selected_features': selected_features,
                'feature_store_results': feature_store_results,
                'feature_stats': self.feature_stats
            }
            
        except Exception as e:
            self.feature_stats['end_time'] = datetime.now()
            error_msg = f"Erreur feature engineering: {e}"
            logger.error(error_msg)
            self.feature_stats['errors'].append(error_msg)
            
            return {
                'status': 'error',
                'error': str(e),
                'feature_stats': self.feature_stats
            }
    
    def _is_holiday_season(self, month):
        """Déterminer si c'est la saison des vacances"""
        return month.isin([11, 12, 1, 7, 8]).astype(int)
    
    def _calculate_text_complexity(self, text):
        """Calculer la complexité du texte"""
        if pd.isna(text) or not text:
            return 0.0
        
        text = str(text)
        words = text.split()
        
        if not words:
            return 0.0
        
        # Complexité basée sur la diversité du vocabulaire et la longueur
        unique_words = len(set(words))
        total_words = len(words)
        
        # Ratio de mots uniques
        uniqueness_ratio = unique_words / total_words
        
        # Longueur moyenne des mots
        avg_word_length = sum(len(word) for word in words) / total_words
        
        # Score de complexité (0-1)
        complexity = (uniqueness_ratio * 0.5 + 
                     min(avg_word_length / 10, 1.0) * 0.3 +
                     min(total_words / 100, 1.0) * 0.2)
        
        return min(complexity, 1.0)
    
    def _calculate_sentiment_intensity(self, text):
        """Calculer l'intensité du sentiment"""
        if pd.isna(text) or not text:
            return 0.0
        
        text = str(text).lower()
        
        # Mots positifs et négatifs simples
        positive_words = ['excellent', 'amazing', 'perfect', 'love', 'great', 'awesome', 'fantastic', 'wonderful']
        negative_words = ['terrible', 'awful', 'hate', 'worst', 'disappointed', 'bad', 'poor', 'horrible']
        
        words = text.split()
        if not words:
            return 0.0
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        # Score d'intensité (-1 à 1)
        if positive_count + negative_count == 0:
            return 0.0
        
        intensity = (positive_count - negative_count) / (positive_count + negative_count)
        return intensity
    
    def generate_feature_report(self, results: Dict):
        """Générer un rapport de feature engineering"""
        report = {
            'feature_engineering_timestamp': datetime.now().isoformat(),
            'results': results,
            'feature_summary': {
                'total_features_created': len(results.get('feature_stats', {}).get('features_created', [])),
                'features_selected': results.get('features_selected', 0),
                'processing_duration': results.get('duration_seconds', 0),
                'records_processed': results.get('records_processed', 0)
            }
        }
        
        # Sauvegarder le rapport
        report_path = "data/processed/feature_engineering_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Rapport de feature engineering sauvegardé: {report_path}")
        return report

def main():
    """Fonction principale du feature engineering"""
    engineer = FeatureEngineer()
    
    # Configuration des sources
    sources = [
        {
            'input_csv': "data/processed/electronics_pandas_transformed_sample.csv",
            'output_base': "data/processed/electronics_features",
            'name': 'Electronics'
        },
        {
            'input_csv': "data/processed/clothing_pandas_transformed_sample.csv",
            'output_base': "data/processed/clothing_features",
            'name': 'Clothing'
        }
    ]
    
    results = []
    
    # Exécuter le feature engineering pour chaque source
    for source in sources:
        print(f"\n{'='*60}")
        print(f"Feature Engineering: {source['name']}")
        print(f"{'='*60}")
        
        result = engineer.run_feature_engineering(
            source['input_csv'],
            source['output_base']
        )
        
        results.append(result)
        
        if result['status'] == 'success':
            print(f"✅ {source['name']}: {result['records_processed']:,} enregistrements")
            print(f"   Features créées: {result['features_created']}")
            print(f"   Features sélectionnées: {result['features_selected']}")
            print(f"   Durée: {result['duration_seconds']:.2f}s")
        else:
            print(f"❌ {source['name']}: {result['error']}")
    
    # Générer le rapport final
    feature_report = engineer.generate_feature_report({'results': results})
    
    print(f"\n{'='*60}")
    print("RAPPORT FEATURE ENGINEERING FINAL")
    print(f"{'='*60}")
    
    total_records = sum(r.get('records_processed', 0) for r in results)
    total_features = sum(r.get('features_created', 0) for r in results)
    total_errors = sum(1 for r in results if r['status'] == 'error')
    
    print(f"Datasets traités: {len(results)}")
    print(f"Total enregistrements: {total_records:,}")
    print(f"Total features créées: {total_features}")
    print(f"Erreurs: {total_errors}")
    
    if total_errors == 0:
        print("✅ Feature engineering terminé avec succès!")
    else:
        print("⚠️  Feature engineering terminé avec des erreurs")

if __name__ == "__main__":
    main()
