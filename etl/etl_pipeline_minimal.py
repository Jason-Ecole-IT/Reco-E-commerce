"""
Pipeline ETL minimal compatible Windows
Focus sur les transformations de base sans configuration Spark avancée
"""

import sys
import os
sys.path.append('src')

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MinimalETLPipeline:
    """Pipeline ETL minimal compatible Windows"""
    
    def __init__(self):
        """Initialiser le pipeline ETL minimal"""
        self.spark = None
        self.pipeline_stats = {
            'start_time': None,
            'end_time': None,
            'records_processed': 0,
            'errors': [],
            'warnings': []
        }
    
    def initialize_spark(self) -> SparkSession:
        """Initialiser Spark avec configuration minimale"""
        try:
            self.spark = SparkSession.builder \
                .appName("minimal_etl_pipeline") \
                .master("local[*]") \
                .getOrCreate()
            
            logger.info("Session Spark initialisée (mode minimal)")
            return self.spark
            
        except Exception as e:
            logger.error(f"Erreur initialisation Spark: {e}")
            raise
    
    def load_source_data(self, source_path: str, sample_size: Optional[int] = None):
        """Charger les données source"""
        try:
            logger.info(f"Chargement des données depuis: {source_path}")
            
            # Charger les données (laisser Spark inférer le schéma)
            df = self.spark.read.json(source_path)
            
            # Échantillonner si nécessaire
            if sample_size:
                total_count = df.count()
                if total_count > sample_size:
                    fraction = sample_size / total_count
                    df = df.sample(False, fraction, seed=42)
                    logger.info(f"Échantillonnage à {sample_size} enregistrements")
            
            initial_count = df.count()
            logger.info(f"Chargé {initial_count:,} enregistrements")
            
            return df, initial_count
            
        except Exception as e:
            error_msg = f"Erreur chargement données: {e}"
            logger.error(error_msg)
            self.pipeline_stats['errors'].append(error_msg)
            raise
    
    def apply_basic_transformations(self, df):
        """Appliquer les transformations de base"""
        try:
            logger.info("Application des transformations de base")
            
            # 1. Nettoyage des champs texte
            df = df.withColumn("reviewText_clean", 
                             F.trim(F.regexp_replace(F.col("reviewText"), "[^\\x20-\\x7E]", "")))
            df = df.withColumn("summary_clean", 
                             F.trim(F.regexp_replace(F.col("summary"), "[^\\x20-\\x7E]", "")))
            
            # 2. Features de base
            df = df.withColumn("word_count", 
                             F.size(F.split(F.col("reviewText_clean"), " ")))
            df = df.withColumn("char_count", 
                             F.length(F.col("reviewText_clean")))
            df = df.withColumn("avg_word_length", 
                             F.when(F.col("word_count") > 0, 
                                    F.col("char_count") / F.col("word_count")).otherwise(0))
            
            # 3. Sentiment basique
            df = df.withColumn("sentiment_score",
                             F.when(F.col("overall") <= 2, -1.0)
                              .when(F.col("overall") == 3, 0.0)
                              .otherwise(1.0))
            
            # 4. Features temporelles
            df = df.withColumn("review_timestamp", 
                             F.to_timestamp(F.col("unixReviewTime")))
            df = df.withColumn("review_date_only", 
                             F.to_date(F.col("review_timestamp")))
            
            # 5. Stats utilisateur
            user_stats = df.groupBy("reviewerID").agg(
                F.count("*").alias("user_review_count"),
                F.avg("overall").alias("user_avg_rating"),
                F.min("review_timestamp").alias("user_first_review"),
                F.max("review_timestamp").alias("user_last_review")
            )
            
            # 6. Stats produit
            product_stats = df.groupBy("asin").agg(
                F.count("*").alias("product_review_count"),
                F.avg("overall").alias("product_avg_rating"),
                F.stddev("overall").alias("product_rating_std")
            )
            
            # 7. Joindre les stats
            df = df.join(user_stats, "reviewerID", "left")
            df = df.join(product_stats, "asin", "left")
            
            # 8. Catégorisations
            df = df.withColumn("user_experience",
                             F.when(F.col("user_review_count") >= 50, "expert")
                              .when(F.col("user_review_count") >= 10, "intermediate")
                              .otherwise("beginner"))
            
            df = df.withColumn("product_popularity",
                             F.when(F.col("product_review_count") >= 100, "high")
                              .when(F.col("product_review_count") >= 10, "medium")
                              .otherwise("low"))
            
            # 9. Features d'engagement
            df = df.withColumn("helpfulness_ratio",
                             F.when(F.col("total_votes") > 0,
                                    F.col("helpful_votes") / F.col("total_votes")).otherwise(0))
            
            # 10. Features de qualité
            df = df.withColumn("review_quality",
                             F.when(F.col("word_count") >= 20, 1.0).otherwise(0.5) *
                             F.when(F.col("helpfulness_ratio") > 0.5, 1.2).otherwise(1.0))
            
            logger.info("Transformations de base appliquées")
            return df
            
        except Exception as e:
            error_msg = f"Erreur transformations: {e}"
            logger.error(error_msg)
            self.pipeline_stats['errors'].append(error_msg)
            raise
    
    def write_processed_data(self, df, table_name: str):
        """Écrire les données transformées en Parquet"""
        try:
            logger.info(f"Écriture des données transformées: {table_name}")
            
            output_path = f"data/processed/{table_name}_minimal"
            
            # Écrire en Parquet avec partitionnement simple
            df.write.mode("overwrite") \
              .partitionBy("review_year") \
              .parquet(output_path)
            
            logger.info(f"Données écrites dans: {output_path}")
            return output_path
            
        except Exception as e:
            error_msg = f"Erreur écriture données: {e}"
            logger.error(error_msg)
            self.pipeline_stats['errors'].append(error_msg)
            raise
    
    def validate_basic_quality(self, df, table_name: str):
        """Validation de qualité basique"""
        try:
            logger.info(f"Validation qualité basique pour: {table_name}")
            
            validation_results = {}
            
            # Validation 1: Comptage de base
            total_records = df.count()
            validation_results['basic_counts'] = {
                'total_records': total_records,
                'unique_users': df.select("reviewerID").distinct().count(),
                'unique_products': df.select("asin").distinct().count()
            }
            
            # Validation 2: Distribution des notes
            rating_dist = df.groupBy("overall").count().orderBy("overall").collect()
            validation_results['rating_distribution'] = {
                str(row['overall']): row['count'] for row in rating_dist
            }
            
            # Validation 3: Features créées
            feature_stats = df.select(
                F.mean("word_count").alias("avg_word_count"),
                F.mean("sentiment_score").alias("avg_sentiment"),
                F.mean("helpfulness_ratio").alias("avg_helpfulness")
            ).collect()[0]
            
            validation_results['feature_stats'] = {
                'avg_word_count': float(feature_stats['avg_word_count']),
                'avg_sentiment': float(feature_stats['avg_sentiment']),
                'avg_helpfulness': float(feature_stats['avg_helpfulness'])
            }
            
            # Validation 4: Distribution des nouvelles catégories
            category_dists = {
                'user_experience': df.groupBy("user_experience").count().collect(),
                'product_popularity': df.groupBy("product_popularity").count().collect()
            }
            
            for cat_name, dist in category_dists.items():
                validation_results[f'{cat_name}_distribution'] = {
                    str(row[cat_name]): row['count'] for row in dist
                }
            
            # Sauvegarder les résultats
            validation_path = f"data/processed/validation_{table_name}_minimal.json"
            with open(validation_path, 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)
            
            logger.info(f"Validation terminée: {validation_path}")
            return validation_results
            
        except Exception as e:
            error_msg = f"Erreur validation qualité: {e}"
            logger.error(error_msg)
            self.pipeline_stats['warnings'].append(error_msg)
            return {}
    
    def run_minimal_etl(self, source_path: str, table_name: str, 
                         sample_size: Optional[int] = None):
        """Exécuter le pipeline ETL minimal"""
        try:
            self.pipeline_stats['start_time'] = datetime.now()
            logger.info(f"Démarrage pipeline ETL minimal pour {table_name}")
            
            # Initialiser Spark
            self.initialize_spark()
            
            # Étape 1: Chargement des données
            df, initial_count = self.load_source_data(source_path, sample_size)
            
            # Étape 2: Transformations de base
            df_transformed = self.apply_basic_transformations(df)
            
            # Étape 3: Validation qualité
            validation_results = self.validate_basic_quality(df_transformed, table_name)
            
            # Étape 4: Écriture des données
            output_path = self.write_processed_data(df_transformed, table_name)
            
            # Statistiques finales
            final_count = df_transformed.count()
            self.pipeline_stats['records_processed'] = final_count
            self.pipeline_stats['end_time'] = datetime.now()
            
            duration = (self.pipeline_stats['end_time'] - 
                       self.pipeline_stats['start_time']).total_seconds()
            
            logger.info(f"Pipeline ETL terminé en {duration:.2f}s")
            logger.info(f"Enregistrements traités: {initial_count:,} -> {final_count:,}")
            
            return {
                'status': 'success',
                'table_name': table_name,
                'output_path': output_path,
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
            
        finally:
            if self.spark:
                self.spark.catalog.clearCache()
                self.spark.stop()
                logger.info("Session Spark arrêtée")

def main():
    """Fonction principale du pipeline ETL minimal"""
    pipeline = MinimalETLPipeline()
    
    # Configuration des sources
    sources = [
        {
            'path': "data/processed/amazon_reviews_electronics_clean.json",
            'table_name': "electronics_minimal",
            'sample_size': 10000
        },
        {
            'path': "data/processed/amazon_reviews_clothing_clean.json", 
            'table_name': "clothing_minimal",
            'sample_size': 8000
        }
    ]
    
    results = []
    
    # Exécuter le pipeline pour chaque source
    for source in sources:
        print(f"\n{'='*60}")
        print(f"Traitement: {source['table_name']}")
        print(f"{'='*60}")
        
        result = pipeline.run_minimal_etl(
            source['path'],
            source['table_name'],
            source['sample_size']
        )
        
        results.append(result)
        
        if result['status'] == 'success':
            print(f"✅ {source['table_name']}: {result['final_count']:,} enregistrements")
            print(f"   Durée: {result['duration_seconds']:.2f}s")
            print(f"   Output: {result['output_path']}")
            
            # Afficher quelques stats
            validation = result['validation_results']
            if 'basic_counts' in validation:
                counts = validation['basic_counts']
                print(f"   Users: {counts['unique_users']:,}")
                print(f"   Products: {counts['unique_products']:,}")
        else:
            print(f"❌ {source['table_name']}: {result['error']}")
    
    # Rapport final
    print(f"\n{'='*60}")
    print("RAPPORT ETL MINIMAL FINAL")
    print(f"{'='*60}")
    
    total_records = sum(r.get('final_count', 0) for r in results)
    total_errors = sum(1 for r in results if r['status'] == 'error')
    
    print(f"Tables traitées: {len(results)}")
    print(f"Total enregistrements: {total_records:,}")
    print(f"Erreurs: {total_errors}")
    
    if total_errors == 0:
        print("✅ Pipeline ETL minimal terminé avec succès!")
    else:
        print("⚠️  Pipeline ETL terminé avec des erreurs")

if __name__ == "__main__":
    main()
