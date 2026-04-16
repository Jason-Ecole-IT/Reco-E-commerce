"""
Dashboard simplifié et moderne - Distribution Notes, Top Produits, Recommandations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Recommandation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration de l'API
API_BASE_URL = os.getenv("API_URL", "http://localhost:8001")

# Styles CSS modernes
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main {
        background: white;
        border-radius: 20px;
        padding: 30px;
        margin: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    h1 {
        color: #667eea;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 30px;
        font-weight: bold;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Titre
st.markdown("<h1>🎯 Dashboard Recommandations</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Configuration")

category = st.sidebar.selectbox(
    "Catégorie",
    ["electronics", "clothing"],
    index=0
)

num_recs = st.sidebar.slider(
    "Nombre de recommandations",
    min_value=5,
    max_value=50,
    value=10,
    step=5
)

if 'current_user_id' not in st.session_state:
    st.session_state.current_user_id = "user_123"

user_id = st.sidebar.text_input(
    "ID Utilisateur",
    value=st.session_state.current_user_id
)

if user_id != st.session_state.current_user_id:
    st.session_state.current_user_id = user_id
    st.session_state.recommendations = None

# Appeler l'API
def call_api(endpoint, params=None):
    try:
        url = f"{API_BASE_URL}{endpoint}"
        response = requests.post(url, json=params, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Erreur API: {e}")
        return None

# Charger les données
@st.cache_data(ttl=300)
def load_data(category):
    try:
        if category == "electronics":
            path = "data/processed/electronics_features_feature_store.csv"
        else:
            path = "data/processed/clothing_features_feature_store.csv"
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Erreur chargement: {e}")
        return None

df = load_data(category)

if df is not None:
    # Distribution des notes
    st.header("📈 Distribution des Notes")
    
    rating_dist = df['overall'].value_counts().sort_index()
    fig_rating = px.bar(
        x=rating_dist.index,
        y=rating_dist.values,
        labels={'x': 'Note', 'y': 'Nombre de Reviews'},
        title="Distribution des Notes",
        color=rating_dist.index,
        color_continuous_scale='viridis'
    )
    fig_rating.update_layout(showlegend=False)
    st.plotly_chart(fig_rating)
    
    # Top produits
    st.header("🏆 Top Produits")
    
    top_products = df.groupby('asin')['overall'].agg(['count', 'mean']).nlargest(10, 'count')
    fig_products = px.bar(
        x=top_products['count'],
        y=top_products.index,
        orientation='h',
        labels={'x': 'Nombre de Reviews', 'y': 'Produit'},
        title="Top 10 Produits par Popularité",
        color=top_products['mean'],
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig_products)
    
    # Section Recommandations
    st.header("🎯 Recommandations Personnalisées")
    
    if st.button("🚀 Générer Recommandations", key=f"generate_{user_id}", type="primary"):
        with st.spinner("Génération des recommandations..."):
            recommendations = call_api(
                "/recommend",
                {
                    "user_id": user_id,
                    "category": category,
                    "num_recommendations": num_recs
                }
            )
            
            if recommendations:
                st.success(f"✅ {len(recommendations['recommendations'])} recommandations pour {user_id}")
                
                recs_df = pd.DataFrame(recommendations['recommendations'])
                recs_df.columns = ['Produit ID', 'Score Popularité', 'Note Moyenne']
                
                st.dataframe(recs_df, hide_index=True)
                
                fig_recs = px.bar(
                    recs_df,
                    x='Score Popularité',
                    y='Produit ID',
                    orientation='h',
                    color='Note Moyenne',
                    color_continuous_scale='RdYlGn',
                    title=f"Recommandations pour {user_id}"
                )
                st.plotly_chart(fig_recs)
                
                if 'processing_time_ms' in recommendations:
                    st.caption(f"⏱️ Temps: {recommendations['processing_time_ms']:.1f}ms")

else:
    st.error("❌ Impossible de charger les données")
    st.info("Assurez-vous que les fichiers feature store existent dans data/processed/")
