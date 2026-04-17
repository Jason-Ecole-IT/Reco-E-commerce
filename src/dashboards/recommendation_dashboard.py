"""
Dashboard Professionnel - Système de Recommandation E-commerce
Design corporate avec KPIs, visualisations avancées et interface professionnelle
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import os

# Configuration de la page
st.set_page_config(
    page_title="Recommendation Engine Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration de l'API
API_BASE_URL = os.getenv("API_URL", "http://localhost:8001")
API_TIMEOUT = 30  # Augmenté à 30s pour l'algorithme hybride

# Styles CSS professionnels
st.markdown("""
<style>
    /* Variables de couleur professionnelles */
    :root {
        --primary-color: #1e3a8a;
        --secondary-color: #3b82f6;
        --accent-color: #f59e0b;
        --success-color: #10b981;
        --danger-color: #ef4444;
        --bg-color: #f8fafc;
        --card-bg: #ffffff;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --border-color: #e2e8f0;
    }

    /* Background principal */
    .stApp {
        background-color: var(--bg-color);
    }

    /* Container principal */
    .main {
        background-color: var(--bg-color);
        padding: 2rem;
    }

    /* En-tête professionnel */
    .header-section {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    .header-title {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        text-align: left;
    }

    .header-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }

    /* Cartes KPI */
    .kpi-card {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }

    .kpi-card:hover {
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }

    .kpi-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0;
    }

    .kpi-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        margin: 0.5rem 0 0 0;
        font-weight: 500;
    }

    .kpi-trend {
        font-size: 0.75rem;
        margin-top: 0.5rem;
    }

    .trend-positive {
        color: var(--success-color);
    }

    .trend-negative {
        color: var(--danger-color);
    }

    /* Sections */
    .section-header {
        color: var(--text-primary);
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--border-color);
    }

    /* Boutons professionnels */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }

    /* Sidebar professionnelle */
    .css-1d391kg {
        background-color: white;
        border-right: 1px solid var(--border-color);
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Footer professionnel */
    .footer {
        background-color: var(--card-bg);
        border-top: 1px solid var(--border-color);
        padding: 1.5rem 2rem;
        margin-top: 3rem;
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.875rem;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .badge-primary {
        background-color: var(--primary-color);
        color: white;
    }

    .badge-success {
        background-color: var(--success-color);
        color: white;
    }

    .badge-warning {
        background-color: var(--accent-color);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# En-tête professionnel
st.markdown("""
<div class="header-section">
    <h1 class="header-title">📊 Recommendation Engine Dashboard</h1>
    <p class="header-subtitle">Système de Recommandation E-commerce - Analytics en Temps Réel</p>
</div>
""", unsafe_allow_html=True)

# Sidebar professionnelle
st.sidebar.markdown("### ⚙️ Configuration")

st.sidebar.markdown("---")

category = st.sidebar.selectbox(
    "📦 Catégorie de Produits",
    ["electronics", "clothing"],
    index=0,
    help="Sélectionnez la catégorie de produits pour les recommandations"
)

st.sidebar.markdown("---")

num_recs = st.sidebar.slider(
    "🎯 Nombre de Recommandations",
    min_value=5,
    max_value=50,
    value=10,
    step=5,
    help="Nombre de produits à recommander"
)

st.sidebar.markdown("---")

if 'current_user_id' not in st.session_state:
    st.session_state.current_user_id = "user_123"

user_id = st.sidebar.text_input(
    "👤 ID Utilisateur",
    value=st.session_state.current_user_id,
    help="Entrez l'ID de l'utilisateur pour des recommandations personnalisées"
)

if user_id != st.session_state.current_user_id:
    st.session_state.current_user_id = user_id
    st.session_state.recommendations = None

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="padding: 1rem; background: #f1f5f9; border-radius: 8px;">
    <strong>ℹ️ Info</strong><br>
    <small>Les recommandations sont générées en temps réel par le moteur ML.</small>
</div>
""", unsafe_allow_html=True)

# Appeler l'API
def call_api(endpoint, data):
    """Appel API avec gestion d'erreurs"""
    try:
        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=data,
            timeout=API_TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error(f"Erreur API: Timeout après {API_TIMEOUT} secondes")
        return None
    except requests.exceptions.ConnectionError as e:
        st.error(f"Erreur API: {e}")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"Erreur API HTTP: {e}")
        return None

# Charger les données
@st.cache_data(ttl=60) # Forcer le rechargement du dashboard en changeant le timeout
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
    # Section KPIs
    st.markdown('<h2 class="section-header">📊 Indicateurs Clés de Performance (KPIs)</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="kpi-card">
            <p class="kpi-value">{:,}</p>
            <p class="kpi-label">Total Reviews</p>
            <p class="kpi-trend trend-positive">📈 Données actives</p>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="kpi-card">
            <p class="kpi-value">{:,}</p>
            <p class="kpi-label">Utilisateurs Uniques</p>
            <p class="kpi-trend trend-positive">👤 Base utilisateurs</p>
        </div>
        """.format(df['reviewerID'].nunique()), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="kpi-card">
            <p class="kpi-value">{:,}</p>
            <p class="kpi-label">Produits Uniques</p>
            <p class="kpi-trend trend-positive">📦 Catalogue</p>
        </div>
        """.format(df['asin'].nunique()), unsafe_allow_html=True)
    
    with col4:
        avg_rating = df['overall'].mean()
        st.markdown("""
        <div class="kpi-card">
            <p class="kpi-value">{:.2f}</p>
            <p class="kpi-label">Note Moyenne</p>
            <p class="kpi-trend trend-positive">⭐ Qualité</p>
        </div>
        """.format(avg_rating), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Distribution des notes
    st.markdown('<h2 class="section-header">📈 Distribution des Notes</h2>', unsafe_allow_html=True)
    
    rating_dist = df['overall'].value_counts().sort_index()
    fig_rating = px.bar(
        x=rating_dist.index,
        y=rating_dist.values,
        labels={'x': 'Note', 'y': 'Nombre de Reviews'},
        title="",
        color=rating_dist.index,
        color_continuous_scale='Blues',
        template='plotly_white'
    )
    fig_rating.update_layout(
        showlegend=False,
        xaxis_title="Note (1-5)",
        yaxis_title="Nombre de Reviews",
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0)
    )
    fig_rating.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=1.5)
    st.plotly_chart(fig_rating, use_container_width=True)
    
    st.markdown("---")
    
    # Top produits
    st.markdown('<h2 class="section-header">🏆 Top Produits par Popularité</h2>', unsafe_allow_html=True)
    
    top_products = df.groupby('asin')['overall'].agg(['count', 'mean']).nlargest(10, 'count')
    fig_products = px.bar(
        x=top_products['count'],
        y=top_products.index,
        orientation='h',
        labels={'x': 'Nombre de Reviews', 'y': 'Produit ID'},
        title="",
        color=top_products['mean'],
        color_continuous_scale='RdYlGn',
        template='plotly_white'
    )
    fig_products.update_layout(
        xaxis_title="Nombre de Reviews",
        yaxis_title="Produit ID",
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=400
    )
    fig_products.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=1.5)
    st.plotly_chart(fig_products, use_container_width=True)
    
    st.markdown("---")
    
    # Section Recommandations
    st.markdown('<h2 class="section-header">🎯 Recommandations Personnalisées</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Générer Recommandations", key=f"generate_{user_id}", type="primary"):
            with st.spinner("Analyse du profil utilisateur et génération des recommandations..."):
                recommendations = call_api(
                    "/recommend",
                    {
                        "user_id": user_id,
                        "category": category,
                        "num_recommendations": num_recs
                    }
                )
                
                if recommendations:
                    st.success(f"✅ {len(recommendations['recommendations'])} recommandations générées pour {user_id}")
                    
                    recs_df = pd.DataFrame(recommendations['recommendations'])
                    recs_df.columns = ['Produit ID', 'Score Popularité', 'Note Moyenne']
                    
                    st.markdown("### 📋 Liste des Recommandations")
                    st.dataframe(recs_df, hide_index=True, use_container_width=True)
                    
                    fig_recs = px.bar(
                        recs_df,
                        x='Score Popularité',
                        y='Produit ID',
                        orientation='h',
                        color='Note Moyenne',
                        color_continuous_scale='RdYlGn',
                        title="",
                        template='plotly_white'
                    )
                    fig_recs.update_layout(
                        xaxis_title="Score de Popularité",
                        yaxis_title="Produit ID",
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=400
                    )
                    fig_recs.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=1.5)
                    st.plotly_chart(fig_recs, use_container_width=True)
                    
                    if 'processing_time_ms' in recommendations:
                        st.caption(f"⏱️ Temps de traitement: {recommendations['processing_time_ms']:.1f}ms")
    
    with col2:
        st.markdown("""
        <div class="kpi-card">
            <h4>ℹ️ Comment ça marche ?</h4>
            <ul style="font-size: 0.9rem; margin-top: 1rem;">
                <li>Analyse de l'historique utilisateur</li>
                <li>Calcul de similarité collaborative</li>
                <li>Ranking par popularité et qualité</li>
                <li>Personnalisation en temps réel</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

else:
    st.error("❌ Impossible de charger les données")
    st.info("Assurez-vous que les fichiers feature store existent dans data/processed/")

# Footer professionnel
st.markdown("""
<div class="footer">
    <p>© 2024 Recommendation Engine Dashboard | Système de Recommandation E-commerce</p>
    <p style="margin-top: 0.5rem; opacity: 0.7;">Powered by Machine Learning & Analytics</p>
</div>
""", unsafe_allow_html=True)
