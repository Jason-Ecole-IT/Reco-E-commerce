"""
Analyse de la qualité des données après nettoyage
Amazon Reviews Electronics + Clothing
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataQualityAnalyzer:
    """Analyseur de qualité des données Amazon Reviews"""
    
    def __init__(self):
        self.quality_metrics = {}
    
    def load_cleaned_data(self, file_path: str) -> pd.DataFrame:
        """
        Charger les données nettoyées
        
        Args:
            file_path: Chemin du fichier JSON nettoyé
            
        Returns:
            DataFrame pandas
        """
        try:
            df = pd.read_json(file_path, lines=True)
            logger.info(f"Chargé {len(df)} enregistrements depuis {file_path}")
            return df
        except Exception as e:
            logger.error(f"Erreur lors du chargement: {e}")
            return pd.DataFrame()
    
    def analyze_data_completeness(self, df: pd.DataFrame, dataset_name: str) -> dict:
        """
        Analyser la complétude des données
        
        Args:
            df: DataFrame à analyser
            dataset_name: Nom du dataset
            
        Returns:
            Dictionnaire de métriques de complétude
        """
        logger.info(f"Analyse de complétude pour {dataset_name}")
        
        completeness = {}
        total_records = len(df)
        
        for column in df.columns:
            non_null_count = df[column].notna().sum()
            completeness_rate = (non_null_count / total_records) * 100
            completeness[column] = {
                'non_null_count': non_null_count,
                'completeness_rate': completeness_rate
            }
        
        # Taux global de complétude
        overall_completeness = np.mean([metrics['completeness_rate'] for metrics in completeness.values()])
        
        return {
            'dataset': dataset_name,
            'total_records': total_records,
            'overall_completeness': overall_completeness,
            'column_completeness': completeness
        }
    
    def analyze_rating_distribution(self, df: pd.DataFrame, dataset_name: str) -> dict:
        """
        Analyser la distribution des notes
        
        Args:
            df: DataFrame à analyser
            dataset_name: Nom du dataset
            
        Returns:
            Dictionnaire de métriques de distribution
        """
        if 'overall' not in df.columns:
            return {}
        
        rating_dist = df['overall'].value_counts().sort_index()
        
        # Statistiques descriptives
        stats = {
            'mean': df['overall'].mean(),
            'median': df['overall'].median(),
            'std': df['overall'].std(),
            'min': df['overall'].min(),
            'max': df['overall'].max()
        }
        
        # Distribution par catégorie
        if 'rating_category' in df.columns:
            category_dist = df['rating_category'].value_counts()
        else:
            category_dist = {}
        
        return {
            'dataset': dataset_name,
            'rating_distribution': rating_dist.to_dict(),
            'rating_stats': stats,
            'category_distribution': category_dist.to_dict() if hasattr(category_dist, 'to_dict') else category_dist
        }
    
    def analyze_user_product_stats(self, df: pd.DataFrame, dataset_name: str) -> dict:
        """
        Analyser les statistiques utilisateurs et produits
        
        Args:
            df: DataFrame à analyser
            dataset_name: Nom du dataset
            
        Returns:
            Dictionnaire de statistiques
        """
        if 'reviewerID' not in df.columns or 'asin' not in df.columns:
            return {}
        
        # Statistiques utilisateurs
        user_stats = df.groupby('reviewerID').agg({
            'overall': ['count', 'mean'],
            'asin': 'nunique'
        }).round(2)
        
        user_stats.columns = ['num_reviews', 'avg_rating', 'num_products']
        
        # Statistiques produits
        product_stats = df.groupby('asin').agg({
            'reviewerID': 'nunique',
            'overall': ['count', 'mean']
        }).round(2)
        
        product_stats.columns = ['num_reviewers', 'num_reviews', 'avg_rating']
        
        # Cold start analysis
        single_review_users = (user_stats['num_reviews'] == 1).sum()
        single_review_products = (product_stats['num_reviews'] == 1).sum()
        
        return {
            'dataset': dataset_name,
            'user_stats': {
                'total_users': len(user_stats),
                'avg_reviews_per_user': user_stats['num_reviews'].mean(),
                'median_reviews_per_user': user_stats['num_reviews'].median(),
                'max_reviews_per_user': user_stats['num_reviews'].max(),
                'single_review_users': single_review_users,
                'single_review_user_rate': (single_review_users / len(user_stats)) * 100
            },
            'product_stats': {
                'total_products': len(product_stats),
                'avg_reviews_per_product': product_stats['num_reviews'].mean(),
                'median_reviews_per_product': product_stats['num_reviews'].median(),
                'max_reviews_per_product': product_stats['num_reviews'].max(),
                'single_review_products': single_review_products,
                'single_review_product_rate': (single_review_products / len(product_stats)) * 100
            }
        }
    
    def analyze_temporal_patterns(self, df: pd.DataFrame, dataset_name: str) -> dict:
        """
        Analyser les patterns temporels
        
        Args:
            df: DataFrame à analyser
            dataset_name: Nom du dataset
            
        Returns:
            Dictionnaire de patterns temporels
        """
        if 'review_date' not in df.columns:
            return {}
        
        # Extraire les composantes temporelles
        df_temp = df.copy()
        
        # S'assurer que review_date est bien en datetime
        if not pd.api.types.is_datetime64_any_dtype(df_temp['review_date']):
            df_temp['review_date'] = pd.to_datetime(df_temp['review_date'])
        
        df_temp['year'] = df_temp['review_date'].dt.year
        df_temp['month'] = df_temp['review_date'].dt.month
        df_temp['dayofweek'] = df_temp['review_date'].dt.dayofweek
        df_temp['quarter'] = df_temp['review_date'].dt.quarter
        
        # Distribution temporelle
        yearly_dist = df_temp['year'].value_counts().sort_index()
        monthly_dist = df_temp['month'].value_counts().sort_index()
        dow_dist = df_temp['dayofweek'].value_counts().sort_index()
        quarterly_dist = df_temp['quarter'].value_counts().sort_index()
        
        # Période couverte
        period_range = {
            'start_date': df_temp['review_date'].min(),
            'end_date': df_temp['review_date'].max(),
            'total_days': (df_temp['review_date'].max() - df_temp['review_date'].min()).days
        }
        
        return {
            'dataset': dataset_name,
            'period_range': period_range,
            'yearly_distribution': yearly_dist.to_dict(),
            'monthly_distribution': monthly_dist.to_dict(),
            'day_of_week_distribution': dow_dist.to_dict(),
            'quarterly_distribution': quarterly_dist.to_dict()
        }
    
    def analyze_text_quality(self, df: pd.DataFrame, dataset_name: str) -> dict:
        """
        Analyser la qualité du texte des reviews
        
        Args:
            df: DataFrame à analyser
            dataset_name: Nom du dataset
            
        Returns:
            Dictionnaire de métriques de qualité texte
        """
        if 'reviewText' not in df.columns:
            return {}
        
        # Statistiques de longueur
        if 'review_length' in df.columns:
            length_stats = {
                'mean_length': df['review_length'].mean(),
                'median_length': df['review_length'].median(),
                'std_length': df['review_length'].std(),
                'min_length': df['review_length'].min(),
                'max_length': df['review_length'].max()
            }
        else:
            df['review_length'] = df['reviewText'].str.len()
            length_stats = {
                'mean_length': df['review_length'].mean(),
                'median_length': df['review_length'].median(),
                'std_length': df['review_length'].std(),
                'min_length': df['review_length'].min(),
                'max_length': df['review_length'].max()
            }
        
        # Word count si disponible
        if 'review_word_count' in df.columns:
            word_stats = {
                'mean_words': df['review_word_count'].mean(),
                'median_words': df['review_word_count'].median(),
                'std_words': df['review_word_count'].std(),
                'min_words': df['review_word_count'].min(),
                'max_words': df['review_word_count'].max()
            }
        else:
            df['review_word_count'] = df['reviewText'].str.split().str.len()
            word_stats = {
                'mean_words': df['review_word_count'].mean(),
                'median_words': df['review_word_count'].median(),
                'std_words': df['review_word_count'].std(),
                'min_words': df['review_word_count'].min(),
                'max_words': df['review_word_count'].max()
            }
        
        return {
            'dataset': dataset_name,
            'length_statistics': length_stats,
            'word_statistics': word_stats
        }
    
    def generate_quality_report(self, df: pd.DataFrame, dataset_name: str) -> dict:
        """
        Générer un rapport complet de qualité
        
        Args:
            df: DataFrame à analyser
            dataset_name: Nom du dataset
            
        Returns:
            Rapport de qualité complet
        """
        logger.info(f"Génération du rapport de qualité pour {dataset_name}")
        
        report = {
            'dataset_name': dataset_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'completeness': self.analyze_data_completeness(df, dataset_name),
            'rating_distribution': self.analyze_rating_distribution(df, dataset_name),
            'user_product_stats': self.analyze_user_product_stats(df, dataset_name),
            'temporal_patterns': self.analyze_temporal_patterns(df, dataset_name),
            'text_quality': self.analyze_text_quality(df, dataset_name)
        }
        
        return report
    
    def visualize_quality_metrics(self, df: pd.DataFrame, dataset_name: str):
        """
        Créer des visualisations des métriques de qualité
        
        Args:
            df: DataFrame à visualiser
            dataset_name: Nom du dataset
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Quality Analysis - {dataset_name}', fontsize=16)
        
        # Distribution des notes
        if 'overall' in df.columns:
            df['overall'].hist(bins=5, ax=axes[0, 0], alpha=0.7)
            axes[0, 0].set_title('Rating Distribution')
            axes[0, 0].set_xlabel('Rating')
            axes[0, 0].set_ylabel('Frequency')
        
        # Distribution des longueurs de reviews
        if 'review_length' in df.columns:
            df['review_length'].hist(bins=50, ax=axes[0, 1], alpha=0.7)
            axes[0, 1].set_title('Review Length Distribution')
            axes[0, 1].set_xlabel('Character Count')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_xlim(0, df['review_length'].quantile(0.95))
        
        # Distribution temporelle (années)
        if 'review_year' in df.columns:
            df['review_year'].hist(bins=20, ax=axes[0, 2], alpha=0.7)
            axes[0, 2].set_title('Reviews by Year')
            axes[0, 2].set_xlabel('Year')
            axes[0, 2].set_ylabel('Frequency')
        
        # Distribution des reviews par utilisateur
        if 'reviewerID' in df.columns:
            user_review_counts = df.groupby('reviewerID').size()
            user_review_counts.hist(bins=50, ax=axes[1, 0], alpha=0.7)
            axes[1, 0].set_title('Reviews per User')
            axes[1, 0].set_xlabel('Number of Reviews')
            axes[1, 0].set_ylabel('Number of Users')
            axes[1, 0].set_xlim(0, user_review_counts.quantile(0.95))
        
        # Distribution des reviews par produit
        if 'asin' in df.columns:
            product_review_counts = df.groupby('asin').size()
            product_review_counts.hist(bins=50, ax=axes[1, 1], alpha=0.7)
            axes[1, 1].set_title('Reviews per Product')
            axes[1, 1].set_xlabel('Number of Reviews')
            axes[1, 1].set_ylabel('Number of Products')
            axes[1, 1].set_xlim(0, product_review_counts.quantile(0.95))
        
        # Catégories de rating
        if 'rating_category' in df.columns:
            df['rating_category'].value_counts().plot(kind='bar', ax=axes[1, 2], alpha=0.7)
            axes[1, 2].set_title('Rating Categories')
            axes[1, 2].set_xlabel('Category')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'data/processed/quality_analysis_{dataset_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_quality_report(self, report: dict, output_path: str):
        """
        Sauvegarder le rapport de qualité
        
        Args:
            report: Rapport de qualité
            output_path: Chemin de sortie
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Rapport de qualité sauvegardé dans: {output_path}")

def main():
    """Fonction principale"""
    analyzer = DataQualityAnalyzer()
    
    # Analyser les données Electronics
    electronics_path = "data/processed/amazon_reviews_electronics_clean.json"
    df_electronics = analyzer.load_cleaned_data(electronics_path)
    
    if not df_electronics.empty:
        print("=== Analyse de Qualité - Electronics ===")
        
        # Générer le rapport
        report_electronics = analyzer.generate_quality_report(df_electronics, "Electronics")
        
        # Afficher les métriques clés
        print(f"\n📊 Métriques Clés Electronics:")
        print(f"  Total records: {report_electronics['completeness']['total_records']:,}")
        print(f"  Overall completeness: {report_electronics['completeness']['overall_completeness']:.1f}%")
        print(f"  Average rating: {report_electronics['rating_distribution']['rating_stats']['mean']:.2f}")
        print(f"  Total users: {report_electronics['user_product_stats']['user_stats']['total_users']:,}")
        print(f"  Total products: {report_electronics['user_product_stats']['product_stats']['total_products']:,}")
        print(f"  Single review users: {report_electronics['user_product_stats']['user_stats']['single_review_user_rate']:.1f}%")
        print(f"  Single review products: {report_electronics['user_product_stats']['product_stats']['single_review_product_rate']:.1f}%")
        
        # Visualisations
        analyzer.visualize_quality_metrics(df_electronics, "Electronics")
        
        # Sauvegarder le rapport
        analyzer.save_quality_report(
            report_electronics, 
            "data/processed/quality_report_electronics.json"
        )
    
    # Analyser les données Clothing
    clothing_path = "data/processed/amazon_reviews_clothing_clean.json"
    df_clothing = analyzer.load_cleaned_data(clothing_path)
    
    if not df_clothing.empty:
        print("\n=== Analyse de Qualité - Clothing ===")
        
        # Générer le rapport
        report_clothing = analyzer.generate_quality_report(df_clothing, "Clothing")
        
        # Afficher les métriques clés
        print(f"\n📊 Métriques Clés Clothing:")
        print(f"  Total records: {report_clothing['completeness']['total_records']:,}")
        print(f"  Overall completeness: {report_clothing['completeness']['overall_completeness']:.1f}%")
        print(f"  Average rating: {report_clothing['rating_distribution']['rating_stats']['mean']:.2f}")
        print(f"  Total users: {report_clothing['user_product_stats']['user_stats']['total_users']:,}")
        print(f"  Total products: {report_clothing['user_product_stats']['product_stats']['total_products']:,}")
        print(f"  Single review users: {report_clothing['user_product_stats']['user_stats']['single_review_user_rate']:.1f}%")
        print(f"  Single review products: {report_clothing['user_product_stats']['product_stats']['single_review_product_rate']:.1f}%")
        
        # Visualisations
        analyzer.visualize_quality_metrics(df_clothing, "Clothing")
        
        # Sauvegarder le rapport
        analyzer.save_quality_report(
            report_clothing, 
            "data/processed/quality_report_clothing.json"
        )
    
    print("\n✅ Analyse de qualité terminée!")

if __name__ == "__main__":
    main()
