"""
Utilitaires pour la création et configuration de sessions Spark
"""

from pyspark.sql import SparkSession
import yaml
from pathlib import Path

def load_config():
    """Charger la configuration depuis config.yaml"""
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError("Fichier config.yaml non trouvé dans configs/")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def create_spark_session(app_name: str = None) -> SparkSession:
    """
    Créer une session Spark avec la configuration optimale
    
    Args:
        app_name: Nom de l'application Spark
        
    Returns:
        SparkSession: Session Spark configurée
    """
    config = load_config()
    spark_config = config['spark']
    
    if app_name is None:
        app_name = spark_config['app_name']
    
    # Configuration Spark optimisée
    builder = SparkSession.builder \
        .appName(app_name) \
        .master(spark_config['master'])
    
    # Ajouter les configurations Spark
    for key, value in spark_config['config'].items():
        builder = builder.config(key, value)
    
    # Créer la session
    spark = builder.getOrCreate()
    
    # Configurer le logging
    spark.sparkContext.setLogLevel("WARN")
    
    print(f"Session Spark créée: {app_name}")
    print(f"Master: {spark.sparkContext.master}")
    print(f"Version Spark: {spark.version}")
    
    return spark

def get_spark_config_summary(spark: SparkSession) -> dict:
    """
    Obtenir un résumé de la configuration Spark
    
    Args:
        spark: Session Spark
        
    Returns:
        dict: Résumé de la configuration
    """
    config = spark.sparkContext.getConf().getAll()
    
    summary = {
        'app_name': spark.sparkContext.appName,
        'master': spark.sparkContext.master,
        'version': spark.version,
        'default_parallelism': spark.sparkContext.defaultParallelism,
        'executor_memory': spark.sparkContext._conf.get('spark.executor.memory', 'N/A'),
        'driver_memory': spark.sparkContext._conf.get('spark.driver.memory', 'N/A'),
        'cores_max': spark.sparkContext._conf.get('spark.cores.max', 'N/A')
    }
    
    return summary
