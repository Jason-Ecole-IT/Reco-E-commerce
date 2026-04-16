import streamlit as st
import pandas as pd
import requests
import json
import random
import altair as alt
from datetime import datetime

# Configuration
st.set_page_config(page_title="E-commerce Recommender", page_icon="🛍️", layout="wide")

# Titre
st.title("🛍️ E-commerce Recommendation System")
st.markdown("Système de recommandation personnalisé pour e-commerce")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Choisir une page", ["Accueil", "Recommandations", "Métriques", "A/B Testing", "À propos"])

# Fonction pour appeler l'API
def call_api(endpoint, data=None):
    try:
        base_url = "http://localhost:8000"
        url = f"{base_url}{endpoint}"

        if data:
            response = requests.post(url, json=data, timeout=30)  # Augmenté à 30 secondes
        else:
            response = requests.get(url, timeout=30)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Erreur de connexion: {e}")
        return None


def load_feature_store(category: str):
    path = f"data/clean_data/features/{category}_features_feature_store.csv"
    return pd.read_csv(path)


def plot_top_reviewed(df: pd.DataFrame, top_n: int = 10):
    product_reviews = df.groupby('asin').agg(review_count=('reviewerID', 'count')).reset_index()
    top_reviewed = product_reviews.sort_values('review_count', ascending=False).head(top_n)
    chart = alt.Chart(top_reviewed).mark_bar().encode(
        x=alt.X('review_count:Q', title='Nombre de reviews'),
        y=alt.Y('asin:N', sort='-x', title='Produit'),
        tooltip=['asin', 'review_count']
    ).properties(height=400)
    return chart


def plot_best_rated(df: pd.DataFrame, top_n: int = 10, min_reviews: int = 20):
    product_ratings = df.groupby('asin').agg(
        avg_rating=('overall', 'mean'),
        review_count=('reviewerID', 'count')
    ).reset_index()
    product_ratings = product_ratings[product_ratings['review_count'] >= min_reviews]
    top_rated = product_ratings.sort_values(['avg_rating', 'review_count'], ascending=[False, False]).head(top_n)
    chart = alt.Chart(top_rated).mark_bar().encode(
        x=alt.X('avg_rating:Q', title='Note moyenne'),
        y=alt.Y('asin:N', sort='-x', title='Produit'),
        color=alt.Color('review_count:Q', title='Nombre de reviews', scale=alt.Scale(scheme='tealblue')),
        tooltip=['asin', alt.Tooltip('avg_rating', format='.2f'), 'review_count']
    ).properties(height=400)
    return chart


def get_popularity_recommendations(df: pd.DataFrame, top_n: int = 10):
    stats = df.groupby('asin').agg(
        review_count=('reviewerID', 'count'),
        avg_rating=('overall', 'mean')
    ).reset_index()
    return stats.sort_values(['review_count', 'avg_rating'], ascending=[False, False]).head(top_n)


def get_ab_state():
    if 'ab_test' not in st.session_state:
        st.session_state.ab_test = {
            'A': {'views': 0, 'clicks': 0},
            'B': {'views': 0, 'clicks': 0}
        }
    return st.session_state.ab_test


def record_ab_event(variant: str, event_type: str):
    ab_state = get_ab_state()
    if variant in ab_state and event_type in ab_state[variant]:
        ab_state[variant][event_type] += 1


if page == "Accueil":
    st.header("Bienvenue")
    st.markdown("""
    Ce système utilise l'intelligence artificielle pour recommander des produits personnalisés
    basés sur les préférences des utilisateurs.
    """)

    # Vérifier la santé de l'API
    if st.button("Vérifier l'état du système"):
        with st.spinner("Vérification en cours..."):
            health = call_api("/health")
            if health:
                st.success("✅ Système opérationnel")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Modèles chargés", health.get("models_loaded", 0))
                with col2:
                    st.metric("Feature stores", health.get("feature_stores_loaded", 0))
                with col3:
                    st.metric("Redis", "✅" if health.get("redis_available") else "❌")
            else:
                st.error("❌ Système indisponible")

elif page == "Recommandations":
    st.header("Obtenir des recommandations")

    col1, col2 = st.columns(2)

    with col1:
        user_id = st.text_input("ID Utilisateur", "A0148968UM59JS3Y8D1M")
        category = st.selectbox("Catégorie", ["clothing", "electronics"])
        num_rec = st.slider("Nombre de recommandations", 1, 10, 5)

    with col2:
        st.markdown("### Exemple d'utilisateur")
        st.code("A0148968UM59JS3Y8D1M")

    if st.button("Obtenir recommandations", type="primary"):
        with st.spinner("Génération des recommandations..."):
            data = {
                "user_id": user_id,
                "category": category,
                "num_recommendations": num_rec
            }

            result = call_api("/recommend", data)
            if result:
                st.success("Recommandations générées !")

                # Afficher les résultats
                for i, rec in enumerate(result.get("recommendations", []), 1):
                    with st.container():
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col1:
                            st.metric(f"#{i}", rec.get("product_id", "N/A"))
                        with col2:
                            st.write(f"Score: {rec.get('score', 0):.2f}")
                        with col3:
                            st.write(rec.get("reason", "N/A"))

elif page == "Métriques":
    st.header("Métriques du système")

    categories = ["clothing", "electronics"]
    selected_category = st.selectbox("Choisir une catégorie", categories)

    try:
        df_store = load_feature_store(selected_category)

        st.subheader(f"Analyse produit - {selected_category.capitalize()}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total avis", len(df_store))
        with col2:
            st.metric("Produits uniques", df_store['asin'].nunique())
        with col3:
            st.metric("Note moyenne", f"{df_store['overall'].mean():.2f}")

        st.markdown("---")

        review_stats = df_store.groupby('asin').agg(
            review_count=('reviewerID', 'count'),
            avg_rating=('overall', 'mean')
        ).reset_index()

        top_reviewed = review_stats.sort_values('review_count', ascending=False).head(10)
        top_rated = review_stats[review_stats['review_count'] >= 20].sort_values(['avg_rating', 'review_count'], ascending=[False, False]).head(10)

        st.subheader("Produits les plus commentés")
        st.altair_chart(
            alt.Chart(top_reviewed).mark_bar().encode(
                x=alt.X('review_count:Q', title='Nombre de reviews'),
                y=alt.Y('asin:N', sort='-x', title='Produit'),
                tooltip=['asin', 'review_count']
            ).properties(height=420),
            use_container_width=True
        )

        st.subheader("Produits les mieux notés")
        st.altair_chart(
            alt.Chart(top_rated).mark_bar().encode(
                x=alt.X('avg_rating:Q', title='Note moyenne'),
                y=alt.Y('asin:N', sort='-x', title='Produit'),
                color=alt.Color('review_count:Q', title='Nombre de reviews', scale=alt.Scale(scheme='teals')),
                tooltip=['asin', alt.Tooltip('avg_rating', format='.2f'), 'review_count']
            ).properties(height=420),
            use_container_width=True
        )

        st.markdown("---")
        st.subheader("Détail top produits")
        st.dataframe(top_reviewed.rename(columns={
            'asin': 'Produit',
            'review_count': 'Nombre de reviews'
        }))

    except FileNotFoundError:
        st.warning("Fichier de feature store introuvable pour la catégorie sélectionnée")
    except Exception as e:
        st.error(f"Erreur lors du chargement des métriques: {e}")

elif page == "A/B Testing":
    st.header("A/B Testing des recommandations")

    exp_name = st.text_input("Nom de l'expérience", "Experiment 1")
    category = st.selectbox("Catégorie", ["clothing", "electronics"], key="ab_category")
    user_id = st.text_input("ID Utilisateur pour test", "A0148968UM59JS3Y8D1M", key="ab_user")
    top_n = st.slider("Taille de la proposition", 1, 10, 5, key="ab_top_n")
    variant_mode = st.radio("Mode A/B", ["Comparaison side-by-side", "Affectation aléatoire"], index=0)

    try:
        df_store = load_feature_store(category)
        popularity_recs = get_popularity_recommendations(df_store, top_n)

        if st.button("Lancer l'expérience A/B"):
            variant = random.choice(["A", "B"]) if variant_mode == "Affectation aléatoire" else "A et B"
            ab_state = get_ab_state()

            if variant == "A":
                record_ab_event("A", "views")
            elif variant == "B":
                record_ab_event("B", "views")
            else:
                record_ab_event("A", "views")
                record_ab_event("B", "views")

            st.success(f"Expérience lancée : variante {variant}")

            if variant_mode == "Affectation aléatoire":
                st.markdown(f"### Variante assignée : {variant}")
            else:
                st.markdown("### Comparaison des variantes")

            st.write("**Variante A — Recommandations collaboratives**")
            recs_a = call_api("/recommend", {"user_id": user_id, "category": category, "num_recommendations": top_n})
            if recs_a and recs_a.get("recommendations"):
                for rec in recs_a["recommendations"]:
                    st.write(f"- {rec['product_id']} — score {rec['score']:.2f}")
            else:
                st.warning("Aucune recommandation collaborative disponible")

            st.write("**Variante B — Recommandations par popularité**")
            for _, row in popularity_recs.iterrows():
                st.write(f"- {row['asin']} — reviews {int(row['review_count'])}, note {row['avg_rating']:.2f}")

            if st.button("Je préfère cette variante A", key="ab_pref_a"):
                record_ab_event("A", "clicks")
                st.success("Feedback enregistré pour la variante A")

            if st.button("Je préfère cette variante B", key="ab_pref_b"):
                record_ab_event("B", "clicks")
                st.success("Feedback enregistré pour la variante B")

        st.markdown("---")
        st.subheader("Résultats A/B")
        ab_state = get_ab_state()
        summary = pd.DataFrame([
            {"Variante": "A", "Vues": ab_state["A"]["views"], "Clics": ab_state["A"]["clicks"], "Taux de conversion": f"{(ab_state['A']['clicks'] / ab_state['A']['views'] * 100) if ab_state['A']['views'] else 0:.1f}%"},
            {"Variante": "B", "Vues": ab_state["B"]["views"], "Clics": ab_state["B"]["clicks"], "Taux de conversion": f"{(ab_state['B']['clicks'] / ab_state['B']['views'] * 100) if ab_state['B']['views'] else 0:.1f}%"}
        ])
        st.dataframe(summary)

        if st.button("Réinitialiser les résultats A/B"):
            st.session_state.ab_test = {
                'A': {'views': 0, 'clicks': 0},
                'B': {'views': 0, 'clicks': 0}
            }
            st.success("Résultats réinitialisés")

    except FileNotFoundError:
        st.warning("Fichier de feature store introuvable pour la catégorie sélectionnée")
    except Exception as e:
        st.error(f"Erreur lors du test A/B: {e}")

elif page == "À propos":
    st.header("À propos")
    st.markdown("""
    ### Architecture technique
    - **ML**: Collaborative Filtering avec SVD
    - **API**: FastAPI avec Redis cache
    - **Features**: 75 features utilisateur/produit
    - **Données**: Amazon reviews (Clothing & Electronics)

    ### Métriques cibles
    - CTR: +15% vs aléatoire
    - Latence API: <50ms
    - Précision@10: >0.15
    """)

    st.markdown("---")
    st.markdown(f"Dernière mise à jour: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Footer
st.markdown("---")
st.markdown("© 2026 - Moteur de recommandation e-commerce")