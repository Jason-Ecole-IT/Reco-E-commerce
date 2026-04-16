"""
Test rapide de PySpark avec les données Amazon Reviews
"""

import sys
import os
sys.path.append('src')

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import time

def test_spark_session():
    """Tester la création de session Spark"""
    try:
        spark = SparkSession.builder \
            .appName("test_amazon_reviews") \
            .master("local[*]") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()
        
        print("Session Spark créée avec succès!")
        print(f"Version Spark: {spark.version}")
        print(f"Master: {spark.sparkContext.master}")
        
        return spark
    except Exception as e:
        print(f"Erreur création session Spark: {e}")
        return None

def test_data_loading(spark):
    """Tester le chargement des données Amazon Reviews"""
    try:
        # Charger les données Electronics
        electronics_path = "data/raw/amazon_reviews_electronics_url.json"
        
        if os.path.exists(electronics_path):
            print(f"\nChargement des données depuis {electronics_path}")
            
            start_time = time.time()
            df = spark.read.json(electronics_path)
            load_time = time.time() - start_time
            
            print(f"Données chargées en {load_time:.2f} secondes")
            print(f"Nombre de reviews: {df.count():,}")
            print(f"Colonnes: {df.columns}")
            
            # Afficher quelques exemples
            print("\nAperçu des données:")
            df.show(3, truncate=False)
            
            # Statistiques rapides
            print("\nStatistiques des notes:")
            df.groupBy("overall").count().orderBy("overall").show()
            
            return True
        else:
            print(f"Fichier non trouvé: {electronics_path}")
            return False
            
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        return False

def main():
    """Fonction principale de test"""
    print("=== Test PySpark avec Amazon Reviews ===")
    
    # Test session Spark
    spark = test_spark_session()
    if not spark:
        return
    
    try:
        # Test chargement données
        if test_data_loading(spark):
            print("\n=== Test réussi! ===")
        else:
            print("\n=== Test échoué ===")
            
    finally:
        spark.stop()
        print("\nSession Spark arrêtée")

if __name__ == "__main__":
    main()
