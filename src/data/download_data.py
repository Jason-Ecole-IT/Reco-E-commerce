"""
Script pour télécharger les datasets Amazon Reviews v2
"""

import os
import gzip
import json
import requests
from pathlib import Path
from tqdm import tqdm
import yaml

def load_config():
    """Charger la configuration depuis config.yaml"""
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

def download_file(url: str, destination: str) -> bool:
    """
    Télécharger un fichier avec barre de progression
    
    Args:
        url: URL du fichier à télécharger
        destination: Chemin de destination
        
    Returns:
        bool: True si succès, False sinon
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f, tqdm(
            desc=f"Téléchargement {Path(destination).name}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"Erreur lors du téléchargement: {e}")
        return False

def extract_gzjson(gz_path: str, json_path: str) -> bool:
    """
    Extraire un fichier JSON compressé en .gz
    
    Args:
        gz_path: Chemin du fichier .gz
        json_path: Chemin de destination du JSON
        
    Returns:
        bool: True si succès, False sinon
    """
    try:
        print(f"Extraction de {gz_path} vers {json_path}")
        
        with gzip.open(gz_path, 'rt', encoding='utf-8') as gz_file:
            with open(json_path, 'w', encoding='utf-8') as json_file:
                for line in tqdm(gz_file, desc="Extraction"):
                    json_file.write(line)
        
        return True
    except Exception as e:
        print(f"Erreur lors de l'extraction: {e}")
        return False

def download_amazon_datasets():
    """Télécharger et extraire les datasets Amazon Reviews"""
    config = load_config()
    data_config = config['data']
    
    # Créer les dossiers
    raw_path = Path(data_config['raw_path'])
    raw_path.mkdir(parents=True, exist_ok=True)
    
    datasets = data_config['amazon_reviews']
    
    for category, url in datasets.items():
        if not url:
            continue
            
        print(f"\n=== Traitement de la catégorie: {category} ===")
        
        # Noms de fichiers
        gz_filename = f"amazon_reviews_{category}.json.gz"
        json_filename = f"amazon_reviews_{category}.json"
        
        gz_path = raw_path / gz_filename
        json_path = raw_path / json_filename
        
        # Téléchargement
        if not gz_path.exists():
            print(f"Téléchargement de {category}...")
            if not download_file(url, str(gz_path)):
                print(f"Échec du téléchargement pour {category}")
                continue
        else:
            print(f"Fichier {gz_filename} déjà existant")
        
        # Extraction
        if not json_path.exists():
            print(f"Extraction de {category}...")
            if not extract_gzjson(str(gz_path), str(json_path)):
                print(f"Échec de l'extraction pour {category}")
                continue
        else:
            print(f"Fichier {json_filename} déjà extrait")
        
        # Afficher la taille
        if json_path.exists():
            size_mb = json_path.stat().st_size / (1024 * 1024)
            print(f"Taille du fichier {json_filename}: {size_mb:.1f} MB")

def main():
    """Fonction principale"""
    print("=== Téléchargement des datasets Amazon Reviews v2 ===")
    download_amazon_datasets()
    print("\n=== Téléchargement terminé ===")

if __name__ == "__main__":
    main()
