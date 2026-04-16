"""
Monitoring qualité simplifié compatible avec les feature stores générés
Validation de base, détection d'anomalies, alerting
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleQualityMonitor:
    """Moniteur de qualité simplifié"""
    
    def __init__(self):
        """Initialiser le moniteur"""
        self.monitoring_stats = {
            'start_time': None,
            'end_time': None,
            'datasets_monitored': [],
            'alerts_generated': []
        }
    
    def load_and_analyze(self, feature_store_path: str, dataset_name: str):
        """Charger et analyser le feature store"""
        try:
            logger.info(f"Analyse du dataset: {dataset_name}")
            
            # Charger les données
            df = pd.read_csv(feature_store_path)
            
            # Statistiques de base
            analysis = {
                'dataset_name': dataset_name,
                'timestamp': datetime.now().isoformat(),
                'basic_stats': {
                    'total_records': len(df),
                    'total_columns': len(df.columns),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
                },
                'data_quality': {},
                'distributions': {},
                'alerts': []
            }
            
            # 1. Validation des colonnes essentielles
            essential_cols = ['reviewerID', 'asin', 'overall']
            missing_essential = [col for col in essential_cols if col not in df.columns]
            
            if missing_essential:
                analysis['alerts'].append({
                    'type': 'MISSING_COLUMNS',
                    'severity': 'HIGH',
                    'message': f"Colonnes essentielles manquantes: {missing_essential}"
                })
            
            # 2. Validation des valeurs nulles
            null_counts = df.isnull().sum()
            high_null_cols = null_counts[null_counts > len(df) * 0.1].index.tolist()
            
            if high_null_cols:
                analysis['alerts'].append({
                    'type': 'HIGH_NULLS',
                    'severity': 'MEDIUM',
                    'message': f"Colonnes avec >10% de valeurs nulles: {high_null_cols[:5]}"
                })
            
            analysis['data_quality']['null_counts'] = null_counts.to_dict()
            
            # 3. Validation des plages de valeurs
            quality_issues = []
            
            if 'overall' in df.columns:
                invalid_ratings = df[~df['overall'].between(1, 5)].shape[0]
                if invalid_ratings > 0:
                    quality_issues.append(f"{invalid_ratings} notes hors plage [1-5]")
            
            # Vérifier les colonnes de ratio
            ratio_cols = [col for col in df.columns if 'ratio' in col.lower()]
            for col in ratio_cols:
                if col in df.columns:
                    invalid_ratios = df[~df[col].between(0, 1)].shape[0]
                    if invalid_ratios > 0:
                        quality_issues.append(f"{invalid_ratios} valeurs invalides dans {col}")
            
            if quality_issues:
                analysis['alerts'].append({
                    'type': 'INVALID_VALUES',
                    'severity': 'HIGH',
                    'message': f"Problèmes de qualité: {'; '.join(quality_issues[:3])}"
                })
            
            # 4. Distribution des notes
            if 'overall' in df.columns:
                rating_dist = df['overall'].value_counts().sort_index().to_dict()
                total_ratings = len(df)
                positive_ratio = sum(rating_dist.get(rating, 0) for rating in [4, 5]) / total_ratings
                
                analysis['distributions']['rating_distribution'] = rating_dist
                analysis['distributions']['positive_ratio'] = positive_ratio
                
                # Vérifier la distribution
                if positive_ratio > 0.9:
                    analysis['alerts'].append({
                        'type': 'BIASED_DISTRIBUTION',
                        'severity': 'MEDIUM',
                        'message': f"Biais positif élevé: {positive_ratio:.2f}"
                    })
            
            # 5. Distribution des utilisateurs
            if 'reviewerID' in df.columns:
                unique_users = df['reviewerID'].nunique()
                total_records = len(df)
                avg_reviews_per_user = total_records / unique_users
                
                analysis['distributions']['user_stats'] = {
                    'unique_users': unique_users,
                    'avg_reviews_per_user': avg_reviews_per_user,
                    'single_review_users': (df['reviewerID'].value_counts() == 1).sum()
                }
                
                # Cold start problem
                single_review_ratio = (df['reviewerID'].value_counts() == 1).sum() / unique_users
                if single_review_ratio > 0.7:
                    analysis['alerts'].append({
                        'type': 'COLD_START',
                        'severity': 'HIGH',
                        'message': f"Problème de cold start: {single_review_ratio:.1%} utilisateurs avec 1 review"
                    })
            
            # 6. Distribution des produits
            if 'asin' in df.columns:
                unique_products = df['asin'].nunique()
                avg_reviews_per_product = total_records / unique_products
                
                analysis['distributions']['product_stats'] = {
                    'unique_products': unique_products,
                    'avg_reviews_per_product': avg_reviews_per_product
                }
            
            # 7. Features numériques
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            analysis['data_quality']['numeric_features_count'] = len(numeric_cols)
            
            # Statistiques descriptives pour les features clés
            key_numeric_cols = ['overall', 'word_count', 'sentiment_score', 'helpfulness_ratio']
            for col in key_numeric_cols:
                if col in df.columns:
                    analysis['data_quality'][f'{col}_stats'] = {
                        'mean': df[col].mean(),
                        'median': df[col].median(),
                        'std': df[col].std(),
                        'min': df[col].min(),
                        'max': df[col].max()
                    }
            
            # 8. Score de qualité global
            score_components = []
            
            # Complétude (-1 si colonnes manquantes)
            if not missing_essential:
                score_components.append(1.0)
            else:
                score_components.append(0.5)
            
            # Distribution des notes (0-1)
            if 'overall' in df.columns and positive_ratio <= 0.8:
                score_components.append(1.0)
            else:
                score_components.append(0.7)
            
            # Volume de données (0-1)
            if len(df) >= 1000:
                score_components.append(1.0)
            else:
                score_components.append(0.5)
            
            # Diversité utilisateurs (0-1)
            if unique_users >= 100:
                score_components.append(1.0)
            else:
                score_components.append(0.7)
            
            overall_score = np.mean(score_components)
            analysis['overall_quality_score'] = overall_score
            
            logger.info(f"Analyse terminée - Score qualité: {overall_score:.2f}")
            return analysis
            
        except Exception as e:
            error_msg = f"Erreur analyse {dataset_name}: {e}"
            logger.error(error_msg)
            return {
                'dataset_name': dataset_name,
                'status': 'error',
                'error': str(e),
                'overall_quality_score': 0.0,
                'alerts': [{'type': 'ERROR', 'severity': 'HIGH', 'message': error_msg}]
            }
    
    def run_monitoring(self, feature_stores: List[Dict]):
        """Exécuter le monitoring sur plusieurs feature stores"""
        try:
            self.monitoring_stats['start_time'] = datetime.now()
            logger.info("Démarrage du monitoring de qualité")
            
            results = []
            
            for store_config in feature_stores:
                result = self.load_and_analyze(
                    store_config['path'], 
                    store_config['name']
                )
                results.append(result)
                
                # Ajouter aux alertes globales
                self.monitoring_stats['alerts_generated'].extend(result.get('alerts', []))
                self.monitoring_stats['datasets_monitored'].append(store_config['name'])
            
            # Finaliser les statistiques
            self.monitoring_stats['end_time'] = datetime.now()
            duration = (self.monitoring_stats['end_time'] - 
                       self.monitoring_stats['start_time']).total_seconds()
            
            # Créer le rapport final
            report = {
                'monitoring_timestamp': datetime.now().isoformat(),
                'duration_seconds': duration,
                'datasets_analyzed': len(feature_stores),
                'results': results,
                'summary': {
                    'total_datasets': len(results),
                    'avg_quality_score': np.mean([r.get('overall_quality_score', 0) for r in results]),
                    'total_alerts': len(self.monitoring_stats['alerts_generated']),
                    'high_severity_alerts': len([a for a in self.monitoring_stats['alerts_generated'] if a.get('severity') == 'HIGH']),
                    'datasets_with_issues': sum(1 for r in results if r.get('overall_quality_score', 0) < 0.7)
                },
                'monitoring_stats': self.monitoring_stats
            }
            
            # Sauvegarder le rapport
            report_path = "data/processed/simple_quality_monitoring_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Monitoring terminé en {duration:.2f}s")
            logger.info(f"Rapport sauvegardé: {report_path}")
            
            return report
            
        except Exception as e:
            error_msg = f"Erreur monitoring: {e}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'error': str(e),
                'monitoring_stats': self.monitoring_stats
            }

def main():
    """Fonction principale du monitoring simplifié"""
    monitor = SimpleQualityMonitor()
    
    # Configuration des feature stores
    feature_stores = [
        {
            'path': "data/processed/electronics_features_feature_store.csv",
            'name': 'Electronics'
        },
        {
            'path': "data/processed/clothing_features_feature_store.csv",
            'name': 'Clothing'
        }
    ]
    
    # Exécuter le monitoring
    report = monitor.run_monitoring(feature_stores)
    
    # Afficher les résultats
    print(f"\n{'='*60}")
    print("RAPPORT MONITORING QUALITÉ")
    print(f"{'='*60}")
    
    if report.get('status') != 'error':
        summary = report['summary']
        print(f"Datasets analysés: {summary['total_datasets']}")
        print(f"Score qualité moyen: {summary['avg_quality_score']:.2f}")
        print(f"Total alertes: {summary['total_alerts']}")
        print(f"Alertes critiques: {summary['high_severity_alerts']}")
        print(f"Datasets avec problèmes: {summary['datasets_with_issues']}")
        
        # Détails par dataset
        print(f"\n{'='*60}")
        print("DÉTAILS PAR DATASET")
        print(f"{'='*60}")
        
        for result in report['results']:
            dataset_name = result['dataset_name']
            quality_score = result.get('overall_quality_score', 0)
            alerts_count = len(result.get('alerts', []))
            
            print(f"\n📊 {dataset_name}:")
            print(f"   Score qualité: {quality_score:.2f}")
            print(f"   Alertes: {alerts_count}")
            
            if result.get('basic_stats'):
                stats = result['basic_stats']
                print(f"   Enregistrements: {stats['total_records']:,}")
                print(f"   Colonnes: {stats['total_columns']}")
                print(f"   Mémoire: {stats['memory_usage_mb']:.1f} MB")
            
            # Afficher les alertes importantes
            critical_alerts = [a for a in result.get('alerts', []) if a.get('severity') == 'HIGH']
            if critical_alerts:
                print(f"   ⚠️  Alertes critiques:")
                for alert in critical_alerts[:3]:
                    print(f"      - {alert['message']}")
            
            # Afficher les distributions clés
            if 'distributions' in result:
                dist = result['distributions']
                
                if 'user_stats' in dist:
                    user_stats = dist['user_stats']
                    print(f"   Utilisateurs: {user_stats['unique_users']:,}")
                    print(f"   Reviews/utilisateur: {user_stats['avg_reviews_per_user']:.1f}")
                
                if 'product_stats' in dist:
                    prod_stats = dist['product_stats']
                    print(f"   Produits: {prod_stats['unique_products']:,}")
                    print(f"   Reviews/produit: {prod_stats['avg_reviews_per_product']:.1f}")
                
                if 'positive_ratio' in dist:
                    print(f"   Ratio notes positives: {dist['positive_ratio']:.2f}")
        
        # Recommandations
        print(f"\n{'='*60}")
        print("RECOMMANDATIONS")
        print(f"{'='*60}")
        
        if summary['avg_quality_score'] < 0.8:
            print("🔧 Améliorer la qualité des données:")
            print("   - Nettoyer les valeurs manquantes")
            print("   - Valider les plages de valeurs")
            print("   - Équilibrer la distribution des notes")
        
        if summary['high_severity_alerts'] > 0:
            print("🚨 Résoudre les problèmes critiques:")
            print("   - Vérifier les colonnes essentielles")
            print("   - Traiter le problème de cold start")
            print("   - Corriger les valeurs invalides")
        
        if summary['datasets_with_issues'] > 0:
            print("📈 Améliorer la couverture:")
            print("   - Augmenter le volume de données")
            print("   - Diversifier les sources")
            print("   - Enrichir les features")
        
        if summary['avg_quality_score'] >= 0.8 and summary['high_severity_alerts'] == 0:
            print("✅ Qualité des données excellente!")
            print("   - Les datasets sont prêts pour la modélisation")
            print("   - Poursuivre avec l'entraînement des modèles")
    
    else:
        print(f"❌ Erreur lors du monitoring: {report.get('error')}")

if __name__ == "__main__":
    main()
