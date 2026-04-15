"""
Monitoring et validation qualité des données avec Great Expectations
Data profiling automatisé, drift monitoring, alerting
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataQualityMonitor:
    """Moniteur de qualité des données avec Great Expectations-like validation"""
    
    def __init__(self):
        """Initialiser le moniteur de qualité"""
        self.quality_stats = {
            'start_time': None,
            'end_time': None,
            'validations_run': [],
            'anomalies_detected': [],
            'alerts_triggered': []
        }
    
    def load_feature_store(self, csv_path: str):
        """Charger le feature store"""
        try:
            logger.info(f"Chargement du feature store: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Conversion des dates si nécessaire
            if 'review_timestamp' in df.columns:
                df['review_timestamp'] = pd.to_datetime(df['review_timestamp'])
            
            logger.info(f"Chargé {len(df):,} enregistrements")
            return df
            
        except Exception as e:
            error_msg = f"Erreur chargement feature store: {e}"
            logger.error(error_msg)
            self.quality_stats['errors'] = self.quality_stats.get('errors', [])
            self.quality_stats['errors'].append(error_msg)
            raise
    
    def profile_data(self, df, dataset_name: str):
        """Profiler les données de manière automatisée"""
        try:
            logger.info(f"Profiling des données: {dataset_name}")
            
            profile = {
                'dataset_name': dataset_name,
                'timestamp': datetime.now().isoformat(),
                'basic_stats': {},
                'data_types': {},
                'missing_values': {},
                'duplicates': {},
                'outliers': {},
                'distributions': {}
            }
            
            # 1. Statistiques de base
            profile['basic_stats'] = {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            }
            
            # 2. Types de données
            profile['data_types'] = df.dtypes.value_counts().to_dict()
            
            # 3. Valeurs manquantes
            missing_counts = df.isnull().sum()
            missing_percentages = (missing_counts / len(df)) * 100
            
            profile['missing_values'] = {
                'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
                'missing_percentages': missing_percentages[missing_percentages > 0].to_dict(),
                'total_missing_cells': missing_counts.sum(),
                'missing_percentage': (missing_counts.sum() / (len(df) * len(df.columns))) * 100
            }
            
            # 4. Doublons
            total_duplicates = df.duplicated().sum()
            user_product_duplicates = df.duplicated(subset=['reviewerID', 'asin']).sum()
            
            profile['duplicates'] = {
                'total_duplicates': total_duplicates,
                'duplicate_percentage': (total_duplicates / len(df)) * 100,
                'user_product_duplicates': user_product_duplicates,
                'user_product_duplicate_percentage': (user_product_duplicates / len(df)) * 100
            }
            
            # 5. Outliers pour les colonnes numériques
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outlier_info = {}
            
            for col in numeric_cols:
                if col in df.columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    
                    outlier_info[col] = {
                        'count': len(outliers),
                        'percentage': (len(outliers) / len(df)) * 100,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'min_value': df[col].min(),
                        'max_value': df[col].max()
                    }
            
            profile['outliers'] = outlier_info
            
            # 6. Distributions pour les colonnes clés
            key_cols = ['overall', 'rating_category', 'user_experience', 'product_popularity']
            distribution_info = {}
            
            for col in key_cols:
                if col in df.columns:
                    if df[col].dtype in ['object', 'category']:
                        distribution_info[col] = df[col].value_counts().to_dict()
                    else:
                        distribution_info[col] = {
                            'mean': df[col].mean(),
                            'median': df[col].median(),
                            'std': df[col].std(),
                            'min': df[col].min(),
                            'max': df[col].max(),
                            'percentiles': {
                                '25%': df[col].quantile(0.25),
                                '50%': df[col].quantile(0.50),
                                '75%': df[col].quantile(0.75),
                                '90%': df[col].quantile(0.90),
                                '95%': df[col].quantile(0.95)
                            }
                        }
            
            profile['distributions'] = distribution_info
            
            logger.info(f"Profiling terminé pour {dataset_name}")
            return profile
            
        except Exception as e:
            error_msg = f"Erreur profiling données: {e}"
            logger.error(error_msg)
            self.quality_stats['errors'] = self.quality_stats.get('errors', [])
            self.quality_stats['errors'].append(error_msg)
            raise
    
    def validate_data_quality(self, df, dataset_name: str):
        """Valider la qualité des données avec règles personnalisées"""
        try:
            logger.info(f"Validation qualité: {dataset_name}")
            
            validation_results = {
                'dataset_name': dataset_name,
                'timestamp': datetime.now().isoformat(),
                'validations': {},
                'overall_score': 0,
                'issues_found': []
            }
            
            total_score = 0
            total_validations = 0
            
            # Validation 1: Complétude des clés
            key_completeness = {
                'reviewerID_not_null': df['reviewerID'].notna().all(),
                'asin_not_null': df['asin'].notna().all(),
                'overall_not_null': df['overall'].notna().all(),
                'score': 1.0 if all([
                    df['reviewerID'].notna().all(),
                    df['asin'].notna().all(),
                    df['overall'].notna().all()
                ]) else 0.0
            }
            
            validation_results['validations']['key_completeness'] = key_completeness
            total_score += key_completeness['score']
            total_validations += 1
            
            if key_completeness['score'] < 1.0:
                validation_results['issues_found'].append("Valeurs nulles dans les clés primaires")
            
            # Validation 2: Plages de valeurs valides
            rating_valid = df['overall'].between(1, 5).all()
            helpful_ratio_valid = df['helpfulness_ratio'].between(0, 1).all()
            
            range_validation = {
                'rating_range_valid': rating_valid,
                'helpful_ratio_range_valid': helpful_ratio_valid,
                'score': 1.0 if rating_valid and helpful_ratio_valid else 0.5
            }
            
            validation_results['validations']['range_validation'] = range_validation
            total_score += range_validation['score']
            total_validations += 1
            
            if not rating_valid:
                validation_results['issues_found'].append("Notes hors plage [1-5]")
            if not helpful_ratio_valid:
                validation_results['issues_found'].append("Ratio helpfulness hors plage [0-1]")
            
            # Validation 3: Distribution des notes
            rating_dist = df['overall'].value_counts().sort_index()
            total_ratings = len(df)
            
            # Vérifier qu'il y a des notes dans chaque catégorie
            has_all_ratings = all(rating in rating_dist.index for rating in [1, 2, 3, 4, 5])
            positive_ratio = (rating_dist.loc[[4, 5]].sum() / total_ratings) if total_ratings > 0 else 0
            
            distribution_validation = {
                'has_all_ratings': has_all_ratings,
                'positive_ratio': positive_ratio,
                'expected_positive_range': (0.3, 0.8),  # 30-80% de notes positives
                'score': 1.0 if has_all_ratings and 0.3 <= positive_ratio <= 0.8 else 0.5
            }
            
            validation_results['validations']['distribution_validation'] = distribution_validation
            total_score += distribution_validation['score']
            total_validations += 1
            
            if not has_all_ratings:
                validation_results['issues_found'].append("Distribution incomplète des notes")
            if not (0.3 <= positive_ratio <= 0.8):
                validation_results['issues_found'].append(f"Ratio de notes positifs anormal: {positive_ratio:.2f}")
            
            # Validation 4: Unicité des reviews
            unique_reviews = df[['reviewerID', 'asin', 'review_timestamp']].drop_duplicates().shape[0]
            uniqueness_ratio = unique_reviews / len(df)
            
            uniqueness_validation = {
                'unique_reviews': unique_reviews,
                'total_reviews': len(df),
                'uniqueness_ratio': uniqueness_ratio,
                'score': min(uniqueness_ratio * 2, 1.0)  # Score basé sur le ratio d'unicité
            }
            
            validation_results['validations']['uniqueness_validation'] = uniqueness_validation
            total_score += uniqueness_validation['score']
            total_validations += 1
            
            if uniqueness_ratio < 0.95:
                validation_results['issues_found'].append(f"Taux d'unicité faible: {uniqueness_ratio:.2f}")
            
            # Validation 5: Qualité du texte
            if 'word_count' in df.columns:
                avg_word_count = df['word_count'].mean()
                min_word_count = df['word_count'].min()
                
                text_quality_validation = {
                    'avg_word_count': avg_word_count,
                    'min_word_count': min_word_count,
                    'score': 1.0 if avg_word_count > 5 and min_word_count > 0 else 0.5
                }
                
                validation_results['validations']['text_quality_validation'] = text_quality_validation
                total_score += text_quality_validation['score']
                total_validations += 1
                
                if avg_word_count <= 5:
                    validation_results['issues_found'].append("Longueur moyenne des reviews trop faible")
                if min_word_count <= 0:
                    validation_results['issues_found'].append("Reviews avec 0 mots")
            
            # Score global
            validation_results['overall_score'] = total_score / total_validations if total_validations > 0 else 0
            
            logger.info(f"Validation terminée - Score global: {validation_results['overall_score']:.2f}")
            return validation_results
            
        except Exception as e:
            error_msg = f"Erreur validation qualité: {e}"
            logger.error(error_msg)
            self.quality_stats['errors'] = self.quality_stats.get('errors', [])
            self.quality_stats['errors'].append(error_msg)
            raise
    
    def detect_data_drift(self, current_df, reference_df, dataset_name: str):
        """Détecter le drift des données par rapport à une référence"""
        try:
            logger.info(f"Détection de drift: {dataset_name}")
            
            drift_results = {
                'dataset_name': dataset_name,
                'timestamp': datetime.now().isoformat(),
                'drift_metrics': {},
                'drift_detected': False,
                'drift_score': 0
            }
            
            # Colonnes numériques à comparer
            numeric_cols = ['overall', 'word_count', 'sentiment_score', 'helpfulness_ratio']
            drift_detected = False
            total_drift_score = 0
            metrics_compared = 0
            
            for col in numeric_cols:
                if col in current_df.columns and col in reference_df.columns:
                    current_stats = current_df[col].describe()
                    ref_stats = reference_df[col].describe()
                    
                    # Test de Kolmogorov-Smirnov pour détecter le drift
                    ks_statistic, ks_pvalue = stats.ks_2samp(
                        current_df[col].dropna(), 
                        reference_df[col].dropna()
                    )
                    
                    # Calcul du drift relatif pour la moyenne
                    mean_drift = abs(current_stats['mean'] - ref_stats['mean']) / ref_stats['mean'] if ref_stats['mean'] != 0 else 0
                    
                    # Calcul du drift relatif pour l'écart-type
                    std_drift = abs(current_stats['std'] - ref_stats['std']) / ref_stats['std'] if ref_stats['std'] != 0 else 0
                    
                    drift_score = (ks_statistic + mean_drift + std_drift) / 3
                    
                    drift_results['drift_metrics'][col] = {
                        'ks_statistic': ks_statistic,
                        'ks_pvalue': ks_pvalue,
                        'mean_drift': mean_drift,
                        'std_drift': std_drift,
                        'drift_score': drift_score,
                        'drift_detected': ks_pvalue < 0.05 or drift_score > 0.1
                    }
                    
                    if drift_results['drift_metrics'][col]['drift_detected']:
                        drift_detected = True
                    
                    total_drift_score += drift_score
                    metrics_compared += 1
            
            # Drift pour les colonnes catégorielles
            categorical_cols = ['rating_category', 'user_experience', 'product_popularity']
            
            for col in categorical_cols:
                if col in current_df.columns and col in reference_df.columns:
                    current_dist = current_df[col].value_counts(normalize=True)
                    ref_dist = reference_df[col].value_counts(normalize=True)
                    
                    # Calculer la distance entre distributions
                    all_categories = set(current_dist.index) | set(ref_dist.index)
                    
                    current_aligned = [current_dist.get(cat, 0) for cat in all_categories]
                    ref_aligned = [ref_dist.get(cat, 0) for cat in all_categories]
                    
                    # Distance de Hellinger
                    hellinger_distance = np.sqrt(sum((np.sqrt(c) - np.sqrt(r))**2 for c, r in zip(current_aligned, ref_aligned))) / np.sqrt(2)
                    
                    drift_detected_cat = hellinger_distance > 0.1
                    
                    drift_results['drift_metrics'][col] = {
                        'hellinger_distance': hellinger_distance,
                        'drift_detected': drift_detected_cat,
                        'drift_score': hellinger_distance
                    }
                    
                    if drift_detected_cat:
                        drift_detected = True
                    
                    total_drift_score += hellinger_distance
                    metrics_compared += 1
            
            # Score global de drift
            drift_results['drift_score'] = total_drift_score / metrics_compared if metrics_compared > 0 else 0
            drift_results['drift_detected'] = drift_detected
            
            logger.info(f"Drift detection terminé - Score: {drift_results['drift_score']:.3f}, Détecté: {drift_results['drift_detected']}")
            return drift_results
            
        except Exception as e:
            error_msg = f"Erreur détection drift: {e}"
            logger.error(error_msg)
            self.quality_stats['errors'] = self.quality_stats.get('errors', [])
            self.quality_stats['errors'].append(error_msg)
            raise
    
    def generate_alerts(self, validation_results, drift_results, dataset_name: str):
        """Générer des alertes basées sur les résultats de validation et drift"""
        try:
            logger.info(f"Génération des alertes: {dataset_name}")
            
            alerts = []
            
            # Alertes de validation
            if validation_results['overall_score'] < 0.7:
                alerts.append({
                    'type': 'QUALITY',
                    'severity': 'HIGH',
                    'message': f"Score de qualité faible: {validation_results['overall_score']:.2f}",
                    'dataset': dataset_name,
                    'timestamp': datetime.now().isoformat()
                })
            
            for issue in validation_results['issues_found']:
                alerts.append({
                    'type': 'QUALITY',
                    'severity': 'MEDIUM',
                    'message': f"Problème de qualité: {issue}",
                    'dataset': dataset_name,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Alertes de drift
            if drift_results['drift_detected']:
                alerts.append({
                    'type': 'DRIFT',
                    'severity': 'HIGH',
                    'message': f"Drift de données détecté - Score: {drift_results['drift_score']:.3f}",
                    'dataset': dataset_name,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Alertes spécifiques par métrique
            for metric, info in drift_results['drift_metrics'].items():
                if info['drift_detected']:
                    severity = 'HIGH' if info['drift_score'] > 0.2 else 'MEDIUM'
                    alerts.append({
                        'type': 'DRIFT',
                        'severity': severity,
                        'message': f"Drift détecté pour {metric}: {info['drift_score']:.3f}",
                        'dataset': dataset_name,
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Alertes de volume
            if 'basic_stats' in validation_results:
                record_count = validation_results['basic_stats'].get('total_records', 0)
                if record_count < 1000:
                    alerts.append({
                        'type': 'VOLUME',
                        'severity': 'MEDIUM',
                        'message': f"Volume de données faible: {record_count:,} enregistrements",
                        'dataset': dataset_name,
                        'timestamp': datetime.now().isoformat()
                    })
            
            logger.info(f"Généré {len(alerts)} alertes pour {dataset_name}")
            return alerts
            
        except Exception as e:
            error_msg = f"Erreur génération alertes: {e}"
            logger.error(error_msg)
            self.quality_stats['errors'] = self.quality_stats.get('errors', [])
            self.quality_stats['errors'].append(error_msg)
            return []
    
    def create_quality_dashboard(self, df, profile, validation, drift, dataset_name: str):
        """Créer un dashboard de qualité des données"""
        try:
            logger.info(f"Création du dashboard: {dataset_name}")
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Data Quality Dashboard - {dataset_name}', fontsize=16)
            
            # 1. Distribution des notes
            if 'overall' in df.columns:
                df['overall'].hist(bins=5, ax=axes[0, 0], alpha=0.7, color='skyblue')
                axes[0, 0].set_title('Rating Distribution')
                axes[0, 0].set_xlabel('Rating')
                axes[0, 0].set_ylabel('Frequency')
            
            # 2. Valeurs manquantes
            if 'missing_values' in profile:
                missing_data = profile['missing_values']['missing_percentages']
                if missing_data:
                    missing_df = pd.DataFrame(list(missing_data.items()), columns=['Column', 'Missing %'])
                    missing_df.plot(kind='bar', x='Column', y='Missing %', ax=axes[0, 1], color='orange')
                    axes[0, 1].set_title('Missing Values by Column')
                    axes[0, 1].set_ylabel('Missing %')
                    axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 3. Score de qualité
            if 'overall_score' in validation:
                score = validation['overall_score']
                colors = ['red' if score < 0.7 else 'orange' if score < 0.9 else 'green']
                axes[0, 2].bar(['Quality Score'], [score], color=colors)
                axes[0, 2].set_title(f'Overall Quality Score: {score:.2f}')
                axes[0, 2].set_ylim(0, 1)
                axes[0, 2].axhline(y=0.7, color='red', linestyle='--', alpha=0.5)
                axes[0, 2].axhline(y=0.9, color='green', linestyle='--', alpha=0.5)
            
            # 4. Distribution des longueurs de reviews
            if 'word_count' in df.columns:
                df['word_count'].hist(bins=50, ax=axes[1, 0], alpha=0.7, color='lightgreen')
                axes[1, 0].set_title('Review Word Count Distribution')
                axes[1, 0].set_xlabel('Word Count')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_xlim(0, df['word_count'].quantile(0.95))
            
            # 5. Drift Score
            if 'drift_score' in drift:
                drift_score = drift['drift_score']
                colors = ['red' if drift_score > 0.1 else 'orange' if drift_score > 0.05 else 'green']
                axes[1, 1].bar(['Drift Score'], [drift_score], color=colors)
                axes[1, 1].set_title(f'Data Drift Score: {drift_score:.3f}')
                axes[1, 1].set_ylim(0, 1)
                axes[1, 1].axhline(y=0.1, color='red', linestyle='--', alpha=0.5)
                axes[1, 1].axhline(y=0.05, color='orange', linestyle='--', alpha=0.5)
            
            # 6. Catégories d'utilisateurs
            if 'user_experience' in df.columns:
                df['user_experience'].value_counts().plot(kind='bar', ax=axes[1, 2], color='purple')
                axes[1, 2].set_title('User Experience Distribution')
                axes[1, 2].set_xlabel('Experience Level')
                axes[1, 2].set_ylabel('Count')
                axes[1, 2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Sauvegarder le dashboard
            dashboard_path = f"data/processed/quality_dashboard_{dataset_name.lower()}.png"
            plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"Dashboard sauvegardé: {dashboard_path}")
            return dashboard_path
            
        except Exception as e:
            error_msg = f"Erreur création dashboard: {e}"
            logger.error(error_msg)
            self.quality_stats['errors'] = self.quality_stats.get('errors', [])
            self.quality_stats['errors'].append(error_msg)
            return None
    
    def run_quality_monitoring(self, feature_store_path: str, reference_path: Optional[str] = None):
        """Exécuter le monitoring complet de qualité"""
        try:
            self.quality_stats['start_time'] = datetime.now()
            logger.info("Démarrage du monitoring de qualité")
            
            # Charger les données
            df = self.load_feature_store(feature_store_path)
            dataset_name = feature_store_path.split('/')[-1].replace('_feature_store.csv', '')
            
            # Profiler les données
            profile = self.profile_data(df, dataset_name)
            
            # Valider la qualité
            validation = self.validate_data_quality(df, dataset_name)
            
            # Détecter le drift (si référence disponible)
            drift = {'drift_score': 0, 'drift_detected': False, 'drift_metrics': {}}
            if reference_path:
                try:
                    ref_df = self.load_feature_store(reference_path)
                    drift = self.detect_data_drift(df, ref_df, dataset_name)
                except Exception as e:
                    logger.warning(f"Impossible de charger les données de référence: {e}")
            
            # Générer les alertes
            alerts = self.generate_alerts(validation, drift, dataset_name)
            
            # Créer le dashboard
            dashboard_path = self.create_quality_dashboard(df, profile, validation, drift, dataset_name)
            
            # Finaliser les statistiques
            self.quality_stats['end_time'] = datetime.now()
            self.quality_stats['validations_run'].append({
                'dataset': dataset_name,
                'timestamp': datetime.now().isoformat(),
                'quality_score': validation['overall_score'],
                'drift_score': drift['drift_score'],
                'alerts_count': len(alerts)
            })
            
            self.quality_stats['anomalies_detected'].extend(alerts)
            self.quality_stats['alerts_triggered'].extend(alerts)
            
            duration = (self.quality_stats['end_time'] - 
                       self.quality_stats['start_time']).total_seconds()
            
            logger.info(f"Monitoring qualité terminé en {duration:.2f}s")
            
            return {
                'status': 'success',
                'dataset_name': dataset_name,
                'duration_seconds': duration,
                'records_processed': len(df),
                'quality_score': validation['overall_score'],
                'drift_score': drift['drift_score'],
                'alerts_count': len(alerts),
                'alerts': alerts,
                'dashboard_path': dashboard_path,
                'profile': profile,
                'validation': validation,
                'drift': drift
            }
            
        except Exception as e:
            self.quality_stats['end_time'] = datetime.now()
            error_msg = f"Erreur monitoring qualité: {e}"
            logger.error(error_msg)
            self.quality_stats['errors'] = self.quality_stats.get('errors', [])
            self.quality_stats['errors'].append(error_msg)
            
            return {
                'status': 'error',
                'error': str(e),
                'quality_stats': self.quality_stats
            }
    
    def generate_monitoring_report(self, results: List[Dict]):
        """Générer un rapport de monitoring complet"""
        report = {
            'monitoring_timestamp': datetime.now().isoformat(),
            'monitoring_results': results,
            'summary': {
                'total_datasets_monitored': len(results),
                'total_records_monitored': sum(r.get('records_processed', 0) for r in results),
                'avg_quality_score': np.mean([r.get('quality_score', 0) for r in results]) if results else 0,
                'avg_drift_score': np.mean([r.get('drift_score', 0) for r in results]) if results else 0,
                'total_alerts': sum(r.get('alerts_count', 0) for r in results),
                'datasets_with_issues': sum(1 for r in results if r.get('quality_score', 0) < 0.7 or r.get('drift_score', 0) > 0.1)
            },
            'quality_stats': self.quality_stats
        }
        
        # Sauvegarder le rapport
        report_path = "data/processed/quality_monitoring_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Rapport de monitoring sauvegardé: {report_path}")
        return report

def main():
    """Fonction principale du monitoring de qualité"""
    monitor = DataQualityMonitor()
    
    # Configuration des feature stores à monitorer
    feature_stores = [
        {
            'path': "data/processed/electronics_features_feature_store.csv",
            'reference': None,  # Pas de référence pour le premier run
            'name': 'Electronics'
        },
        {
            'path': "data/processed/clothing_features_feature_store.csv",
            'reference': None,
            'name': 'Clothing'
        }
    ]
    
    results = []
    
    # Exécuter le monitoring pour chaque feature store
    for store in feature_stores:
        print(f"\n{'='*60}")
        print(f"Monitoring Qualité: {store['name']}")
        print(f"{'='*60}")
        
        result = monitor.run_quality_monitoring(
            store['path'],
            store['reference']
        )
        
        results.append(result)
        
        if result['status'] == 'success':
            print(f"Qualité: {result['quality_score']:.2f}")
            print(f"Drift: {result['drift_score']:.3f}")
            print(f"Alertes: {result['alerts_count']}")
            print(f"Dashboard: {result['dashboard_path']}")
            
            # Afficher les alertes critiques
            critical_alerts = [a for a in result['alerts'] if a['severity'] == 'HIGH']
            if critical_alerts:
                print(f"Alertes critiques: {len(critical_alerts)}")
                for alert in critical_alerts[:3]:  # Afficher les 3 premières
                    print(f"  - {alert['message']}")
        else:
            print(f"Erreur: {result['error']}")
    
    # Générer le rapport final
    monitoring_report = monitor.generate_monitoring_report(results)
    
    print(f"\n{'='*60}")
    print("RAPPORT MONITORING QUALITÉ FINAL")
    print(f"{'='*60}")
    
    summary = monitoring_report['summary']
    print(f"Datasets monitorés: {summary['total_datasets_monitored']}")
    print(f"Total enregistrements: {summary['total_records_monitored']:,}")
    print(f"Score qualité moyen: {summary['avg_quality_score']:.2f}")
    print(f"Score drift moyen: {summary['avg_drift_score']:.3f}")
    print(f"Total alertes: {summary['total_alerts']}")
    print(f"Datasets avec problèmes: {summary['datasets_with_issues']}")
    
    if summary['datasets_with_issues'] == 0:
        print("Monitoring qualité terminé sans problèmes critiques!")
    else:
        print("Attention: Des problèmes de qualité ont été détectés.")

if __name__ == "__main__":
    main()
