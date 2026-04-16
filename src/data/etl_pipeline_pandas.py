"""
Pipeline ETL avec Pandas (compatible Windows)
Focus sur les transformations et la qualité des données sans dépendances Spark
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import re
import os

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PandasETLPipeline:
    """Pipeline ETL avec Pandas pour compatibilité Windows"""
    
    def __init__(self):
        """Initialiser le pipeline ETL"""
        self.pipeline_stats = {
            'start_time': None,
            'end_time': None,
            'records_processed': 0,
            'errors': [],
            'warnings': []
        }
    
    def load_source_data(self, source_path: str, sample_size: Optional[int] = None):
        """Charger les données source avec Pandas"""
        try:
            logger.info(f"Chargement des données depuis: {source_path}")
            
            # Charger les données JSON lines
            data = []
            with open(source_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if sample_size and i >= sample_size:
                        break
                    try:
                        record = json.loads(line.strip())
                        data.append(record)
                    except json.JSONDecodeError:
                        continue
            
            df = pd.DataFrame(data)
            initial_count = len(df)
            logger.info(f"Chargé {initial_count:,} enregistrements")
            
            return df, initial_count
            
        except Exception as e:
            error_msg = f"Erreur chargement données: {e}"
            logger.error(error_msg)
            self.pipeline_stats['errors'].append(error_msg)
            raise
    
    def clean_text(self, text):
        """Nettoyer le texte"""
        if pd.isna(text):
            return ""
        
        text = str(text).strip()
        text = re.sub(r'[^\x20-\x7E]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def apply_data_transformations(self, df):
        """Appliquer les transformations avec Pandas"""
        try:
            logger.info("Application des transformations de données")
            
            # 1. Nettoyer les champs texte
            df['reviewText_clean'] = df['reviewText'].apply(self.clean_text)
            df['summary_clean'] = df['summary'].apply(self.clean_text)
            
            # 2. Features de base
            df['word_count'] = df['reviewText_clean'].str.split().str.len()
            df['char_count'] = df['reviewText_clean'].str.len()
            df['avg_word_length'] = np.where(
                df['word_count'] > 0,
                df['char_count'] / df['word_count'],
                0
            )
            
            # 3. Sentiment basique
            def get_sentiment(rating):
                if rating <= 2:
                    return -1.0
                elif rating == 3:
                    return 0.0
                else:
                    return 1.0
            
            df['sentiment_score'] = df['overall'].apply(get_sentiment)
            
            # 4. Normaliser les notes (Z-score)
            rating_mean = df['overall'].mean()
            rating_std = df['overall'].std()
            df['rating_zscore'] = (df['overall'] - rating_mean) / rating_std
            
            # 5. Features temporelles
            df['review_timestamp'] = pd.to_datetime(df['unixReviewTime'], unit='s')
            df['review_date_only'] = df['review_timestamp'].dt.date
            df['days_since_epoch'] = (datetime.now() - df['review_timestamp']).dt.days
            
            # 6. Stats utilisateur
            user_stats = df.groupby('reviewerID').agg({
                'overall': ['count', 'mean', 'std'],
                'review_timestamp': ['min', 'max']
            }).round(2)
            
            user_stats.columns = ['user_review_count', 'user_avg_rating', 
                                'user_rating_std', 'user_first_review', 'user_last_review']
            
            # 7. Stats produit
            product_stats = df.groupby('asin').agg({
                'overall': ['count', 'mean', 'std']
            }).round(2)
            
            product_stats.columns = ['product_review_count', 'product_avg_rating', 
                                 'product_rating_std']
            
            # 8. Joindre les stats
            df = df.merge(user_stats, left_on='reviewerID', right_index=True, how='left')
            df = df.merge(product_stats, left_on='asin', right_index=True, how='left')
            
            # 9. Catégorisations
            df['user_experience'] = np.where(
                df['user_review_count'] >= 50, 'expert',
                np.where(df['user_review_count'] >= 10, 'intermediate', 'beginner')
            )
            
            df['product_popularity'] = np.where(
                df['product_review_count'] >= 100, 'high',
                np.where(df['product_review_count'] >= 10, 'medium', 'low')
            )
            
            # 10. Features d'engagement
            df['helpfulness_ratio'] = np.where(
                df['total_votes'] > 0,
                df['helpful_votes'] / df['total_votes'],
                0
            )
            
            # 11. Features de qualité
            df['review_quality'] = np.where(
                df['word_count'] >= 20, 1.0, 0.5
            ) * np.where(
                df['helpfulness_ratio'] > 0.5, 1.2, 1.0
            ) * np.where(
                df['rating_category'] == 'positive', 1.1, 1.0
            )
            
            # 12. Features d'interaction
            df['interaction_strength'] = (
                df['overall'] * 
                df['helpfulness_ratio'] * 
                np.log(df['user_review_count'] + 1)
            )
            
            logger.info("Transformations appliquées avec succès")
            return df
            
        except Exception as e:
            error_msg = f"Erreur transformations: {e}"
            logger.error(error_msg)
            self.pipeline_stats['errors'].append(error_msg)
            raise
    
    def validate_data_quality(self, df, table_name: str):
        """Valider la qualité des données transformées"""
        try:
            logger.info(f"Validation qualité pour: {table_name}")
            
            validation_results = {}
            
            # Validation 1: Comptage de base
            validation_results['basic_counts'] = {
                'total_records': len(df),
                'unique_users': df['reviewerID'].nunique(),
                'unique_products': df['asin'].nunique(),
                'null_reviewerID': df['reviewerID'].isnull().sum(),
                'null_asin': df['asin'].isnull().sum(),
                'null_overall': df['overall'].isnull().sum()
            }
            
            # Validation 2: Distribution des notes
            rating_dist = df['overall'].value_counts().sort_index().to_dict()
            validation_results['rating_distribution'] = rating_dist
            
            # Validation 3: Features créées
            validation_results['feature_stats'] = {
                'avg_word_count': df['word_count'].mean(),
                'avg_sentiment': df['sentiment_score'].mean(),
                'avg_helpfulness': df['helpfulness_ratio'].mean(),
                'avg_quality': df['review_quality'].mean(),
                'avg_interaction': df['interaction_strength'].mean()
            }
            
            # Validation 4: Distribution des nouvelles catégories
            validation_results['category_distributions'] = {
                'user_experience': df['user_experience'].value_counts().to_dict(),
                'product_popularity': df['product_popularity'].value_counts().to_dict(),
                'rating_category': df['rating_category'].value_counts().to_dict()
            }
            
            # Validation 5: Plages de valeurs
            validation_results['value_ranges'] = {
                'overall_range': [df['overall'].min(), df['overall'].max()],
                'zscore_range': [df['rating_zscore'].min(), df['rating_zscore'].max()],
                'sentiment_range': [df['sentiment_score'].min(), df['sentiment_score'].max()],
                'helpfulness_range': [df['helpfulness_ratio'].min(), df['helpfulness_ratio'].max()]
            }
            
            # Validation 6: Corrélations basiques
            numeric_cols = ['overall', 'word_count', 'sentiment_score', 'helpfulness_ratio']
            correlation_matrix = df[numeric_cols].corr()
            validation_results['correlations'] = correlation_matrix.to_dict()
            
            # Sauvegarder les résultats
            validation_path = f"data/processed/validation_{table_name}_pandas.json"
            with open(validation_path, 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)
            
            logger.info(f"Validation terminée: {validation_path}")
            return validation_results
            
        except Exception as e:
            error_msg = f"Erreur validation qualité: {e}"
            logger.error(error_msg)
            self.pipeline_stats['warnings'].append(error_msg)
            return {}
    
    def write_processed_data(self, df, table_name: str):
        """Écrire les données transformées en CSV/Parquet"""
        try:
            logger.info(f"Écriture des données transformées: {table_name}")
            
            # Créer le dossier de sortie
            output_dir = "data/processed"
            os.makedirs(output_dir, exist_ok=True)
            
            # Échantillonner pour éviter des fichiers trop grands
            sample_size = min(10000, len(df))
            df_sample = df.sample(n=sample_size, random_state=42)
            
            # Écrire en CSV (compatible partout)
            csv_path = f"{output_dir}/{table_name}_transformed_sample.csv"
            df_sample.to_csv(csv_path, index=False, encoding='utf-8')
            
            # Écrire aussi en JSON pour compatibilité
            json_path = f"{output_dir}/{table_name}_transformed_sample.json"
            df_sample.to_json(json_path, orient='records', lines=True, date_format='iso')
            
            logger.info(f"Données écrites: CSV ({csv_path}), JSON ({json_path})")
            return csv_path, json_path
            
        except Exception as e:
            error_msg = f"Erreur écriture données: {e}"
            logger.error(error_msg)
            self.pipeline_stats['errors'].append(error_msg)
            raise
    
    def run_etl_pipeline(self, source_path: str, table_name: str, 
                       sample_size: Optional[int] = None):
        """Exécuter le pipeline ETL complet"""
        try:
            self.pipeline_stats['start_time'] = datetime.now()
            logger.info(f"Démarrage pipeline ETL pour {table_name}")
            
            # Étape 1: Chargement des données
            df, initial_count = self.load_source_data(source_path, sample_size)
            
            # Étape 2: Transformations
            df_transformed = self.apply_data_transformations(df)
            
            # Étape 3: Validation qualité
            validation_results = self.validate_data_quality(df_transformed, table_name)
            
            # Étape 4: Écriture des données
            csv_path, json_path = self.write_processed_data(df_transformed, table_name)
            
            # Statistiques finales
            final_count = len(df_transformed)
            self.pipeline_stats['records_processed'] = final_count
            self.pipeline_stats['end_time'] = datetime.now()
            
            duration = (self.pipeline_stats['end_time'] - 
                       self.pipeline_stats['start_time']).total_seconds()
            
            logger.info(f"Pipeline ETL terminé en {duration:.2f}s")
            logger.info(f"Enregistrements traités: {initial_count:,} -> {final_count:,}")
            
            return {
                'status': 'success',
                'table_name': table_name,
                'csv_path': csv_path,
                'json_path': json_path,
                'initial_count': initial_count,
                'final_count': final_count,
                'duration_seconds': duration,
                'validation_results': validation_results,
                'pipeline_stats': self.pipeline_stats
            }
            
        except Exception as e:
            self.pipeline_stats['end_time'] = datetime.now()
            error_msg = f"Erreur pipeline ETL: {e}"
            logger.error(error_msg)
            self.pipeline_stats['errors'].append(error_msg)
            
            return {
                'status': 'error',
                'error': str(e),
                'pipeline_stats': self.pipeline_stats
            }
    
    def generate_etl_report(self, results: List[Dict]):
        """Générer un rapport ETL complet"""
        report = {
            'etl_run_timestamp': datetime.now().isoformat(),
            'pipeline_results': results,
            'summary': {
                'total_tables_processed': len(results),
                'total_records_processed': sum(r.get('final_count', 0) for r in results),
                'total_errors': sum(1 for r in results if r['status'] == 'error'),
                'total_warnings': sum(len(r.get('pipeline_stats', {}).get('warnings', [])) for r in results)
            }
        }
        
        # Sauvegarder le rapport
        report_path = "data/processed/etl_report_pandas.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Rapport ETL sauvegardé: {report_path}")
        return report

def main():
    """Fonction principale du pipeline ETL"""
    pipeline = PandasETLPipeline()
    
    # Configuration des sources
    sources = [
        {
            'path': "data/processed/amazon_reviews_electronics_clean.json",
            'table_name': "electronics_pandas",
            'sample_size': 15000
        },
        {
            'path': "data/processed/amazon_reviews_clothing_clean.json", 
            'table_name': "clothing_pandas",
            'sample_size': 10000
        }
    ]
    
    results = []
    
    # Exécuter le pipeline pour chaque source
    for source in sources:
        print(f"\n{'='*60}")
        print(f"Traitement: {source['table_name']}")
        print(f"{'='*60}")
        
        result = pipeline.run_etl_pipeline(
            source['path'],
            source['table_name'],
            source['sample_size']
        )
        
        results.append(result)
        
        if result['status'] == 'success':
            print(f"✅ {source['table_name']}: {result['final_count']:,} enregistrements")
            print(f"   Durée: {result['duration_seconds']:.2f}s")
            print(f"   Output CSV: {result['csv_path']}")
            print(f"   Output JSON: {result['json_path']}")
            
            # Afficher quelques stats
            validation = result['validation_results']
            if 'basic_counts' in validation:
                counts = validation['basic_counts']
                print(f"   Users: {counts['unique_users']:,}")
                print(f"   Products: {counts['unique_products']:,}")
                
            if 'feature_stats' in validation:
                stats = validation['feature_stats']
                print(f"   Avg word count: {stats['avg_word_count']:.1f}")
                print(f"   Avg sentiment: {stats['avg_sentiment']:.2f}")
        else:
            print(f"❌ {source['table_name']}: {result['error']}")
    
    # Générer le rapport final
    etl_report = pipeline.generate_etl_report(results)
    
    print(f"\n{'='*60}")
    print("RAPPORT ETL PANDAS FINAL")
    print(f"{'='*60}")
    print(f"Tables traitées: {etl_report['summary']['total_tables_processed']}")
    print(f"Total enregistrements: {etl_report['summary']['total_records_processed']:,}")
    print(f"Erreurs: {etl_report['summary']['total_errors']}")
    print(f"Avertissements: {etl_report['summary']['total_warnings']}")
    
    if etl_report['summary']['total_errors'] == 0:
        print("✅ Pipeline ETL Pandas terminé avec succès!")
    else:
        print("⚠️  Pipeline ETL terminé avec des erreurs")

if __name__ == "__main__":
    main()
