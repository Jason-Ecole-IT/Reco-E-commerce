import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime

# Configuration
st.set_page_config(page_title="E-commerce Recommender", page_icon="🛍️", layout="wide")

# Titre
st.title("🛍️ E-commerce Recommendation System")
st.markdown("Système de recommandation personnalisé pour e-commerce")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Choisir une page", ["Accueil", "Recommandations", "Métriques", "À propos"])

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

    # Charger les recommandations depuis le fichier
    try:
        df = pd.read_csv("data/output/recommendations_sklearn.csv")
        st.subheader("Aperçu des recommandations")
        st.dataframe(df.head(20))

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Utilisateurs avec recommandations", df['user_id'].nunique())
        with col2:
            st.metric("Total recommandations", len(df))

    except FileNotFoundError:
        st.warning("Fichier de recommandations non trouvé")

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