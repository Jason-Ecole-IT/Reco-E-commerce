
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 1. CONFIG
# =========================
DATA_PATH = Path("data/clean_data/amazon_reviews_clothing_clean.json")
MAX_ROWS = 100000  # mets None pour tout charger si ta machine supporte
MIN_USER_INTERACTIONS = 3
MIN_ITEM_INTERACTIONS = 3
N_COMPONENTS = 20
TOP_N = 5
OUTPUT_DIR = Path("data/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. CHARGER LE JSON
# =========================
def load_json_lines(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


df = load_json_lines(DATA_PATH, MAX_ROWS)

print("Colonnes disponibles :", df.columns.tolist())
print("Nombre de lignes brutes :", len(df))

# =========================
# 3. GARDER LES BONNES COLONNES
# reviewerID = user
# asin = produit
# overall = rating
# =========================
df = df[["reviewerID", "asin", "overall"]].rename(
    columns={
        "reviewerID": "user_id",
        "asin": "product_id",
        "overall": "rating",
    }
)

# Nettoyage
df = df.dropna().drop_duplicates()
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df = df.dropna(subset=["rating"])

print("Nombre de lignes après nettoyage :", len(df))
print(df.head())

# =========================
# 4. FILTRAGE
# On garde seulement les users/items avec assez d'interactions
# =========================
user_counts = df["user_id"].value_counts()
item_counts = df["product_id"].value_counts()

df = df[
    df["user_id"].isin(user_counts[user_counts >= MIN_USER_INTERACTIONS].index)
]
df = df[
    df["product_id"].isin(item_counts[item_counts >= MIN_ITEM_INTERACTIONS].index)
]

print("Nombre de lignes après filtrage :", len(df))
print("Nombre d'utilisateurs :", df["user_id"].nunique())
print("Nombre de produits :", df["product_id"].nunique())

if df.empty:
    raise ValueError("Le dataset est vide après filtrage. Réduis MIN_USER_INTERACTIONS ou MIN_ITEM_INTERACTIONS.")

# =========================
# 5. MATRICE USER-PRODUIT
# lignes = utilisateurs
# colonnes = produits
# valeurs = ratings
# =========================
user_item_matrix = df.pivot_table(
    index="user_id",
    columns="product_id",
    values="rating",
    fill_value=0
)

print("Shape matrice user-item :", user_item_matrix.shape)

# =========================
# 6. REDUCTION DIMENSIONNELLE
# TruncatedSVD = approximation des préférences latentes
# =========================
n_components = min(N_COMPONENTS, max(2, min(user_item_matrix.shape) - 1))
svd = TruncatedSVD(n_components=n_components, random_state=42)

user_factors = svd.fit_transform(user_item_matrix)
item_factors = svd.components_.T  # produits dans l'espace latent

print("Shape user_factors :", user_factors.shape)
print("Shape item_factors :", item_factors.shape)

# =========================
# 7. SIMILARITE ENTRE PRODUITS
# =========================
item_similarity = cosine_similarity(item_factors)
item_ids = user_item_matrix.columns.tolist()

item_similarity_df = pd.DataFrame(
    item_similarity,
    index=item_ids,
    columns=item_ids,
)

# =========================
# 8. FONCTION DE RECOMMANDATION
# Logique :
# - regarder les produits déjà notés par l'utilisateur
# - prendre les produits similaires
# - calculer un score pondéré
# =========================
def recommend_for_user(user_id: str, top_n: int = 5) -> list[dict]:
    if user_id not in user_item_matrix.index:
        # cold start : top produits populaires
        popular = (
            df.groupby("product_id")
            .agg(
                avg_rating=("rating", "mean"),
                interaction_count=("rating", "count"),
            )
            .sort_values(["interaction_count", "avg_rating"], ascending=False)
            .head(top_n)
            .reset_index()
        )
        return [
            {
                "product_id": row["product_id"],
                "score": float(row["avg_rating"]),
                "reason": "popular_product_cold_start",
            }
            for _, row in popular.iterrows()
        ]

    user_ratings = user_item_matrix.loc[user_id]
    already_seen = user_ratings[user_ratings > 0].index.tolist()

    scores = {}

    for product in already_seen:
        rating = user_ratings[product]
        similar_products = item_similarity_df[product].sort_values(ascending=False)

        for similar_product, sim_score in similar_products.items():
            if similar_product == product:
                continue
            if similar_product in already_seen:
                continue

            scores[similar_product] = scores.get(similar_product, 0.0) + (sim_score * rating)

    if not scores:
        popular = (
            df.groupby("product_id")
            .agg(
                avg_rating=("rating", "mean"),
                interaction_count=("rating", "count"),
            )
            .sort_values(["interaction_count", "avg_rating"], ascending=False)
            .head(top_n)
            .reset_index()
        )
        return [
            {
                "product_id": row["product_id"],
                "score": float(row["avg_rating"]),
                "reason": "fallback_popular_products",
            }
            for _, row in popular.iterrows()
        ]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return [
        {
            "product_id": product_id,
            "score": float(score),
            "reason": "similar_to_user_history",
        }
        for product_id, score in ranked
    ]


# =========================
# 9. TEST SUR UN UTILISATEUR
# =========================
sample_user = user_item_matrix.index[0]
recommendations = recommend_for_user(sample_user, top_n=TOP_N)

print(f"\nRecommandations pour user_id={sample_user}")
for rec in recommendations:
    print(rec)

# =========================
# 10. SAUVEGARDE DE QUELQUES RECOMMANDATIONS
# =========================
sample_users = user_item_matrix.index[:50]
rows = []

for user_id in sample_users:
    recs = recommend_for_user(user_id, top_n=TOP_N)
    for rank, rec in enumerate(recs, start=1):
        rows.append({
            "user_id": user_id,
            "rank": rank,
            "product_id": rec["product_id"],
            "score": rec["score"],
            "reason": rec["reason"],
        })

recommendations_df = pd.DataFrame(rows)
output_file = OUTPUT_DIR / "recommendations_sklearn.csv"
recommendations_df.to_csv(output_file, index=False, encoding="utf-8")

print(f"\nFichier sauvegardé : {output_file}")
print(recommendations_df.head(10))