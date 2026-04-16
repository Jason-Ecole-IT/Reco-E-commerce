"""
Test d'exploration des données avec Pandas (alternative à PySpark pour le début)
"""

import pandas as pd
import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def load_amazon_reviews_sample(file_path, max_lines=10000):
    """
    Charger un échantillon des données Amazon Reviews
    
    Args:
        file_path: Chemin du fichier JSON
        max_lines: Nombre maximum de lignes à charger
        
    Returns:
        DataFrame pandas avec les données
    """
    data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                    
        df = pd.DataFrame(data)
        print(f"Chargé {len(df)} reviews depuis {file_path}")
        return df
        
    except Exception as e:
        print(f"Erreur lors du chargement: {e}")
        return None

def explore_data(df, dataset_name):
    """
    Explorer les données et afficher les statistiques
    
    Args:
        df: DataFrame pandas
        dataset_name: Nom du dataset
    """
    print(f"\n=== Exploration du dataset: {dataset_name} ===")
    
    # Info de base
    print(f"Nombre de reviews: {len(df):,}")
    print(f"Colonnes: {list(df.columns)}")
    
    # Statistiques des notes
    if 'overall' in df.columns:
        print(f"\nDistribution des notes:")
        rating_dist = df['overall'].value_counts().sort_index()
        print(rating_dist)
        
        # Visualisation
        plt.figure(figsize=(10, 6))
        plt.bar(rating_dist.index, rating_dist.values)
        plt.title(f'Distribution des notes - {dataset_name}')
        plt.xlabel('Note')
        plt.ylabel('Nombre de reviews')
        plt.show()
    
    # Statistiques utilisateurs
    if 'reviewerID' in df.columns:
        unique_users = df['reviewerID'].nunique()
        reviews_per_user = df.groupby('reviewerID').size()
        
        print(f"\nStatistiques utilisateurs:")
        print(f"Nombre d'utilisateurs uniques: {unique_users:,}")
        print(f"Moyenne de reviews par utilisateur: {reviews_per_user.mean():.2f}")
        print(f"Max reviews par utilisateur: {reviews_per_user.max()}")
        print(f"Utilisateurs avec 1 review: {(reviews_per_user == 1).sum():,}")
    
    # Statistiques produits
    if 'asin' in df.columns:
        unique_products = df['asin'].nunique()
        reviews_per_product = df.groupby('asin').size()
        
        print(f"\nStatistiques produits:")
        print(f"Nombre de produits uniques: {unique_products:,}")
        print(f"Moyenne de reviews par produit: {reviews_per_product.mean():.2f}")
        print(f"Max reviews par produit: {reviews_per_product.max()}")
        print(f"Produits avec 1 review: {(reviews_per_product == 1).sum():,}")
    
    # Valeurs manquantes
    print(f"\nValeurs manquantes:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    # Analyse temporelle
    if 'unixReviewTime' in df.columns:
        df['review_date'] = pd.to_datetime(df['unixReviewTime'], unit='s')
        df['year'] = df['review_date'].dt.year
        
        print(f"\nPériode couverte:")
        print(f"Première review: {df['review_date'].min()}")
        print(f"Dernière review: {df['review_date'].max()}")
        print(f"Reviews par année:")
        yearly_counts = df['year'].value_counts().sort_index()
        print(yearly_counts)

def main():
    """Fonction principale"""
    print("=== Test d'exploration avec Pandas ===")
    
    # Test avec les données Electronics (échantillon)
    electronics_path = "data/raw/amazon_reviews_electronics_url.json"
    
    if os.path.exists(electronics_path):
        print("Chargement des données Electronics...")
        df_electronics = load_amazon_reviews_sample(electronics_path, 50000)
        
        if df_electronics is not None:
            explore_data(df_electronics, "Electronics")
    else:
        print(f"Fichier non trouvé: {electronics_path}")
    
    # Test avec les données Clothing (échantillon)
    clothing_path = "data/raw/amazon_reviews_clothing_url.json"
    
    if os.path.exists(clothing_path):
        print("\nChargement des données Clothing...")
        df_clothing = load_amazon_reviews_sample(clothing_path, 50000)
        
        if df_clothing is not None:
            explore_data(df_clothing, "Clothing")
    else:
        print(f"Fichier non trouvé: {clothing_path}")

if __name__ == "__main__":
    main()
