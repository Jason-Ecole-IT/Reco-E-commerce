"""
Pipeline ETL simplifié sans Delta Lake (compatible Windows)
Focus sur les transformations et la qualité des données
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
from typing import Dict, List, Optional, Tuple
import yaml
from utils.spark_session import create_spark_session

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleETLPipeline:
    """Pipeline ETL simplifié avec focus sur les transformations"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialiser le pipeline ETL"""
        self.config = self._load_config(config_path)
        self.spark = None
        self.pipeline_stats = {
            'start_time': None,
            'end_time': None,
            'records_processed': 0,
            'errors': [],
            'warnings': []
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """Charger la configuration YAML"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Erreur chargement config: {e}")
            return {}
    
    def initialize_spark(self) -> SparkSession:
        """Initialiser Spark sans Delta Lake"""
        try:
            self.spark = create_spark_session("simple_etl_pipeline")
            
            # Configuration optimisée (sans Delta)
            spark_config = {
                "spark.sql.adaptive.enabled": "true",
                "spark.sql.adaptive.coalescePartitions.enabled": "true",
                "spark.sql.adaptive.advisoryPartitionSizeInBytes": "128MB",
                "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
                "spark.sql.execution.arrow.pyspark.enabled": "true",
                "spark.sql.shuffle.partitions": "200",
                "spark.default.parallelism": "200",
                "spark.sql.autoBroadcastJoinThreshold": "10MB"
            }
            
            for key, value in spark_config.items():
                self.spark.conf.set(key, value)
            
            logger.info("Session Spark initialisée (mode simplifié)")
            return self.spark
            
        except Exception as e:
            logger.error(f"Erreur initialisation Spark: {e}")
            raise
    
    def create_udfs(self):
        """Créer les UDFs optimisées pour le pipeline"""
        
        # UDF pour nettoyer le texte
        def clean_text_udf(text):
            if text is None:
                return ""
            
            import re
            text = str(text).strip()
            text = re.sub(r'[^\x20-\x7E]', '', text)
            text = re.sub(r'\s+', ' ', text)
            return text
        
        # UDF pour calculer le sentiment basique
        def sentiment_score_udf(rating):
            if rating is None:
                return 0.0
            if rating <= 2:
                return -1.0
            elif rating == 3:
                return 0.0
            else:
                return 1.0
        
        # UDF pour extraire des features du texte
        def extract_text_features_udf(text):
            if text is None:
                return (0, 0, 0.0)
            
            text = str(text)
            words = text.split()
            word_count = len(words)
            char_count = len(text)
            avg_word_length = char_count / word_count if word_count > 0 else 0.0
            
            return (word_count, char_count, avg_word_length)
        
        # UDF pour calculer le helpfulness score
        def helpfulness_score_udf(helpful_votes, total_votes):
            if total_votes is None or total_votes == 0:
                return 0.0
            ratio = helpful_votes / total_votes
            # Pondérer par le nombre total de votes
            return ratio * min(total_votes / 10.0, 1.0)
        
        # Enregistrer les UDFs
        self.clean_text_udf = F.udf(clean_text_udf, StringType())
        self.sentiment_score_udf = F.udf(sentiment_score_udf, DoubleType())
        self.extract_features_udf = F.udf(extract_text_features_udf, 
                                        StructType([
                                            StructField("word_count", IntegerType(), False),
                                            StructField("char_count", IntegerType(), False),
                                            StructField("avg_word_length", DoubleType(), False)
                                        ]))
        self.helpfulness_score_udf = F.udf(helpfulness_score_udf, DoubleType())
        
        logger.info("UDFs créés et enregistrés")
    
    def load_source_data(self, source_path: str, sample_size: Optional[int] = None):
        """Charger les données source avec gestion d'erreurs"""
        try:
            logger.info(f"Chargement des données depuis: {source_path}")
            
            # Définir le schéma optimisé
            schema = StructType([
                StructField("reviewerID", StringType(), False),
                StructField("asin", StringType(), False),
                StructField("reviewerName", StringType(), True),
                StructField("helpful", ArrayType(IntegerType()), True),
                StructField("reviewText", StringType(), True),
                StructField("overall", DoubleType(), False),
                StructField("summary", StringType(), True),
                StructField("unixReviewTime", LongType(), False),
                StructField("reviewTime", StringType(), True),
                StructField("review_length", IntegerType(), True),
                StructField("review_word_count", IntegerType(), True),
                StructField("helpful_votes", IntegerType(), True),
                StructField("total_votes", IntegerType(), True),
                StructField("helpful_ratio", DoubleType(), True),
                StructField("review_date", TimestampType(), True),
                StructField("review_year", IntegerType(), True),
                StructField("review_month", IntegerType(), True),
                StructField("review_dayofweek", IntegerType(), True),
                StructField("rating_category", StringType(), True)
            ])
            
            # Charger les données
            df = self.spark.read.json(source_path, schema=schema)
            
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
    
    def apply_data_transformations(self, df):
        """Appliquer les transformations avec UDFs optimisées"""
        try:
            logger.info("Application des transformations de données")
            
            # 1. Nettoyer les champs texte
            df = df.withColumn("reviewText_clean", self.clean_text_udf(F.col("reviewText")))
            df = df.withColumn("summary_clean", self.clean_text_udf(F.col("summary")))
            
            # 2. Extraire des features du texte
            df = df.withColumn("text_features", self.extract_features_udf(F.col("reviewText_clean")))
            df = df.withColumn("word_count", F.col("text_features.word_count"))
            df = df.withColumn("char_count", F.col("text_features.char_count"))
            df = df.withColumn("avg_word_length", F.col("text_features.avg_word_length"))
            df = df.drop("text_features")
            
            # 3. Calculer le sentiment
            df = df.withColumn("sentiment_score", self.sentiment_score_udf(F.col("overall")))
            
            # 4. Normaliser les notes (Z-score)
            rating_stats = df.select(F.mean("overall").alias("mean"), 
                                 F.stddev("overall").alias("std")).collect()[0]
            
            df = df.withColumn("rating_zscore", 
                             (F.col("overall") - rating_stats["mean"]) / rating_stats["std"])
            
            # 5. Features temporelles avancées
            df = df.withColumn("review_timestamp", F.to_timestamp(F.col("unixReviewTime")))
            df = df.withColumn("days_since_epoch", 
                             F.datediff(F.current_date(), F.to_date("review_timestamp")))
            
            # 6. Features de récence et fréquence
            window_user = Window.partitionBy("reviewerID").orderBy(F.desc("review_timestamp"))
            df = df.withColumn("user_review_rank", F.row_number().over(window_user))
            df = df.withColumn("user_total_reviews", F.count("*").over(window_user.rangeBetween(-sys.maxsize, 0)))
            
            # 7. Features d'engagement améliorées
            df = df.withColumn("helpfulness_score", 
                             self.helpfulness_score_udf(F.col("helpful_votes"), F.col("total_votes")))
            
            # 8. Catégoriser les produits par popularité
            product_stats = df.groupBy("asin").agg(
                F.count("*").alias("product_review_count"),
                F.avg("overall").alias("product_avg_rating"),
                F.stddev("overall").alias("product_rating_std")
            )
            
            # Joindre les stats produits
            df = df.join(F.broadcast(product_stats), "asin", "left")
            
            # Catégoriser la popularité
            df = df.withColumn("product_popularity",
                             F.when(F.col("product_review_count") >= 100, "high")
                              .when(F.col("product_review_count") >= 10, "medium")
                              .otherwise("low"))
            
            # 9. Features utilisateur avancées
            user_stats = df.groupBy("reviewerID").agg(
                F.count("*").alias("user_review_count"),
                F.avg("overall").alias("user_avg_rating"),
                F.stddev("overall").alias("user_rating_std"),
                F.min("review_timestamp").alias("user_first_review"),
                F.max("review_timestamp").alias("user_last_review")
            )
            
            # Joindre les stats utilisateurs
            df = df.join(F.broadcast(user_stats), "reviewerID", "left")
            
            # Calculer la durée d'activité utilisateur
            df = df.withColumn("user_activity_days",
                             F.datediff(F.col("user_last_review"), 
                                       F.col("user_first_review")))
            
            # Catégoriser l'expérience utilisateur
            df = df.withColumn("user_experience",
                             F.when(F.col("user_review_count") >= 50, "expert")
                              .when(F.col("user_review_count") >= 10, "intermediate")
                              .otherwise("beginner"))
            
            # 10. Features de qualité des reviews
            df = df.withColumn("review_quality_score",
                             F.when(F.col("word_count") >= 50, 1.0).otherwise(0.5) *
                             F.when(F.col("helpfulness_score") > 0.5, 1.2).otherwise(1.0) *
                             F.when(F.col("rating_category") == "positive", 1.1).otherwise(1.0))
            
            # 11. Features d'interaction
            df = df.withColumn("user_item_interaction_strength",
                             F.col("overall") * F.col("helpfulness_score") * 
                             F.log(F.col("user_review_count") + 1))
            
            logger.info("Transformations appliquées avec succès")
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
            
            output_path = f"data/processed/{table_name}_transformed"
            
            # Écrire en Parquet avec partitionnement
            df.write.mode("overwrite") \
              .partitionBy("review_year", "rating_category") \
              .parquet(output_path)
            
            # Créer une vue temporaire
            df.createOrReplaceTempView(f"vw_{table_name}")
            
            logger.info(f"Données écrites dans: {output_path}")
            return output_path
            
        except Exception as e:
            error_msg = f"Erreur écriture données: {e}"
            logger.error(error_msg)
            self.pipeline_stats['errors'].append(error_msg)
            raise
    
    def validate_data_quality(self, df, table_name: str):
        """Valider la qualité des données transformées"""
        try:
            logger.info(f"Validation qualité pour: {table_name}")
            
            validation_results = {}
            
            # Validation 1: Complétude des clés
            key_completeness = {
                'total_records': df.count(),
                'reviewerID_not_null': df.filter(F.col("reviewerID").isNotNull()).count(),
                'asin_not_null': df.filter(F.col("asin").isNotNull()).count(),
                'overall_not_null': df.filter(F.col("overall").isNotNull()).count()
            }
            validation_results['key_completeness'] = key_completeness
            
            # Validation 2: Plages de valeurs
            range_validation = {
                'rating_range_valid': df.filter(
                    (F.col("overall") >= 1) & (F.col("overall") <= 5)
                ).count(),
                'zscore_in_range': df.filter(
                    (F.col("rating_zscore") >= -3) & (F.col("rating_zscore") <= 3)
                ).count(),
                'positive_helpfulness': df.filter(
                    (F.col("helpfulness_score") >= 0) & (F.col("helpfulness_score") <= 10)
                ).count()
            }
            validation_results['range_validation'] = range_validation
            
            # Validation 3: Distribution des nouvelles features
            feature_distribution = {
                'sentiment_distribution': df.groupBy("sentiment_score").count().orderBy("sentiment_score").collect(),
                'popularity_distribution': df.groupBy("product_popularity").count().collect(),
                'experience_distribution': df.groupBy("user_experience").count().collect()
            }
            validation_results['feature_distribution'] = feature_distribution
            
            # Validation 4: Statistiques descriptives des nouvelles features
            descriptive_stats = df.select(
                F.mean("rating_zscore").alias("zscore_mean"),
                F.stddev("rating_zscore").alias("zscore_std"),
                F.mean("helpfulness_score").alias("helpfulness_mean"),
                F.mean("review_quality_score").alias("quality_mean"),
                F.mean("user_item_interaction_strength").alias("interaction_mean")
            ).collect()[0]
            
            validation_results['descriptive_stats'] = {
                'rating_zscore': {'mean': descriptive_stats['zscore_mean'], 'std': descriptive_stats['zscore_std']},
                'helpfulness_score': {'mean': descriptive_stats['helpfulness_mean']},
                'review_quality_score': {'mean': descriptive_stats['quality_mean']},
                'interaction_strength': {'mean': descriptive_stats['interaction_mean']}
            }
            
            # Sauvegarder les résultats de validation
            validation_path = f"data/processed/validation_{table_name}.json"
            with open(validation_path, 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)
            
            logger.info(f"Validation terminée: {validation_path}")
            return validation_results
            
        except Exception as e:
            error_msg = f"Erreur validation qualité: {e}"
            logger.error(error_msg)
            self.pipeline_stats['warnings'].append(error_msg)
            return {}
    
    def run_etl_pipeline(self, source_path: str, table_name: str, 
                       sample_size: Optional[int] = None):
        """Exécuter le pipeline ETL complet"""
        try:
            self.pipeline_stats['start_time'] = datetime.now()
            logger.info(f"Démarrage pipeline ETL pour {table_name}")
            
            # Initialiser Spark
            self.initialize_spark()
            self.create_udfs()
            
            # Étape 1: Chargement des données
            df, initial_count = self.load_source_data(source_path, sample_size)
            
            # Étape 2: Transformations
            df_transformed = self.apply_data_transformations(df)
            
            # Étape 3: Validation qualité
            validation_results = self.validate_data_quality(df_transformed, table_name)
            
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
    
    def generate_etl_report(self, results: List[Dict]):
        """Générer un rapport ETL complet"""
        report = {
            'etl_run_timestamp': datetime.now().isoformat(),
            'pipeline_results': results,
            'summary': {
                'total_tables_processed': len(results),
                'total_records_processed': sum(r.get('final_count', 0) for r in results),
                'total_errors': sum(len(r.get('pipeline_stats', {}).get('errors', [])) for r in results),
                'total_warnings': sum(len(r.get('pipeline_stats', {}).get('warnings', [])) for r in results)
            }
        }
        
        # Sauvegarder le rapport
        report_path = "data/processed/etl_report_simple.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Rapport ETL sauvegardé: {report_path}")
        return report

def main():
    """Fonction principale du pipeline ETL"""
    pipeline = SimpleETLPipeline()
    
    # Configuration des sources
    sources = [
        {
            'path': "data/processed/amazon_reviews_electronics_clean.json",
            'table_name': "amazon_reviews_electronics",
            'sample_size': 30000
        },
        {
            'path': "data/processed/amazon_reviews_clothing_clean.json", 
            'table_name': "amazon_reviews_clothing",
            'sample_size': 20000
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
            print(f"   Output: {result['output_path']}")
        else:
            print(f"❌ {source['table_name']}: {result['error']}")
    
    # Générer le rapport final
    etl_report = pipeline.generate_etl_report(results)
    
    print(f"\n{'='*60}")
    print("RAPPORT ETL FINAL")
    print(f"{'='*60}")
    print(f"Tables traitées: {etl_report['summary']['total_tables_processed']}")
    print(f"Total enregistrements: {etl_report['summary']['total_records_processed']:,}")
    print(f"Erreurs: {etl_report['summary']['total_errors']}")
    print(f"Avertissements: {etl_report['summary']['total_warnings']}")
    
    if etl_report['summary']['total_errors'] == 0:
        print("✅ Pipeline ETL terminé avec succès!")
    else:
        print("⚠️  Pipeline ETL terminé avec des erreurs")

if __name__ == "__main__":
    main()
