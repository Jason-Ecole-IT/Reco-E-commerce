"""
Dashboard Professionnel A/B Testing - Système d'Expérimentation
Design corporate avec KPIs, visualisations avancées et interface professionnelle
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime

sys.path.append('.')

from src.ab_testing.ab_framework import ABTestingFramework, ABTestConfig, TestGroup

# Configuration de la page
st.set_page_config(
    page_title="A/B Testing Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        --warning-color: #f59e0b;
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

    /* Box de succès */
    .success-box {
        background-color: #d1fae5;
        border: 1px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #065f46;
        font-weight: 500;
    }

    /* Box de warning */
    .warning-box {
        background-color: #fef3c7;
        border: 1px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #92400e;
        font-weight: 500;
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
        background-color: var(--warning-color);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# En-tête professionnel
st.markdown("""
<div class="header-section">
    <h1 class="header-title">🧪 A/B Testing Dashboard</h1>
    <p class="header-subtitle">Système d'Expérimentation Statistique - Analytics en Temps Réel</p>
</div>
""", unsafe_allow_html=True)

# Initialiser le framework A/B testing
if 'ab_framework' not in st.session_state:
    st.session_state.ab_framework = ABTestingFramework()
    st.session_state.ab_framework.active_tests = {}

if 'data_generated' not in st.session_state:
    st.session_state.data_generated = {}

ab_framework = st.session_state.ab_framework

# Bouton pour réinitialiser le framework (debug)
if st.sidebar.button("🔄 Réinitialiser Framework", help="Réinitialiser le framework A/B testing"):
    st.session_state.ab_framework = ABTestingFramework()
    st.session_state.ab_framework.active_tests = {}
    st.session_state.data_generated = {}
    st.success("✅ Framework réinitialisé")
    st.rerun()

# Sidebar professionnelle
st.sidebar.markdown("### ⚙️ Configuration")
st.sidebar.markdown("---")

# Créer un nouveau test
with st.sidebar.expander("➕ Créer Nouveau Test", expanded=False):
    test_name = st.text_input("Nom du Test", "Recommendation CTR Test", help="Nom unique du test A/B")
    description = st.text_area("Description", "Test CTR des recommandations ML vs Random", help="Description détaillée du test")
    traffic_split = st.slider("Split Traffic (A/B)", 0.1, 0.9, 0.5, 0.1, help="Pourcentage de trafic vers le groupe A")
    min_sample_size = st.number_input("Sample Size Min", 100, 10000, 1000, 100, help="Taille d'échantillon minimale pour la significativité")
    
    metrics = st.multiselect(
        "Métriques à tracker",
        ["ctr", "conversion_rate", "avg_session_duration", "revenue_per_user"],
        default=["ctr", "conversion_rate"],
        help="Métriques à suivre pendant le test"
    )
    
    if st.button("🚀 Créer Test", type="primary"):
        config = ABTestConfig(
            test_name=test_name,
            description=description,
            start_date=datetime.now(),
            traffic_split=traffic_split,
            min_sample_size=min_sample_size,
            metrics=metrics
        )
        
        try:
            test_id = ab_framework.create_test(config)
            st.success(f"✅ Test créé: {test_id}")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Erreur: {e}")

st.sidebar.markdown("---")

# Liste des tests actifs
st.sidebar.markdown("### 📋 Tests Actifs")

if ab_framework.active_tests:
    test_ids = list(ab_framework.active_tests.keys())
    selected_test = st.sidebar.selectbox("Sélectionner un Test", test_ids, help="Sélectionnez le test à analyser")
    
    if selected_test:
        test_data = ab_framework.active_tests[selected_test]
        config = test_data["config"]
        
        st.sidebar.markdown(f"""
        <div style="padding: 1rem; background: #f1f5f9; border-radius: 8px; margin-top: 1rem;">
            <strong>📊 {config.test_name}</strong><br>
            <small>
            Status: <span class="badge badge-success">{test_data['status']}</span><br>
            Split: {config.traffic_split * 100}% / {(1 - config.traffic_split) * 100}%<br>
            Sample Size: {config.min_sample_size}<br>
            Assignments: {len(test_data['assignments'])}
            </small>
        </div>
        """, unsafe_allow_html=True)
else:
    st.sidebar.warning("⚠️ Aucun test actif")
    selected_test = None

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="padding: 1rem; background: #f1f5f9; border-radius: 8px;">
    <strong>ℹ️ Info</strong><br>
    <small>Les tests A/B permettent de comparer deux versions pour optimiser les conversions.</small>
</div>
""", unsafe_allow_html=True)

# Section principale
if selected_test:
    test_data = ab_framework.active_tests[selected_test]
    config = test_data["config"]
    
    # En-tête du test
    st.markdown('<h2 class="section-header">📊 Détails du Test</h2>', unsafe_allow_html=True)
    st.markdown(f"**{config.test_name}**")
    st.markdown(f"<small>{config.description}</small>", unsafe_allow_html=True)
    
    # KPIs du test
    col1, col2, col3, col4 = st.columns(4)
    
    group_a_count = sum(1 for g in test_data["assignments"].values() if g == TestGroup.CONTROL)
    group_b_count = sum(1 for g in test_data["assignments"].values() if g == TestGroup.TREATMENT)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-value">{test_data['status']}</p>
            <p class="kpi-label">Status du Test</p>
            <p class="kpi-trend trend-positive">📊 État actuel</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-value">{len(test_data['assignments'])}</p>
            <p class="kpi-label">Total Assignments</p>
            <p class="kpi-trend trend-positive">👤 Utilisateurs assignés</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-value">{group_a_count}</p>
            <p class="kpi-label">Group A (Control)</p>
            <p class="kpi-trend trend-positive">🔵 Groupe de contrôle</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-value">{group_b_count}</p>
            <p class="kpi-label">Group B (Treatment)</p>
            <p class="kpi-trend trend-positive">🟠 Groupe test</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Simulation de données pour démonstration
    st.markdown('<h2 class="section-header">🎮 Simulation de Données</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        num_simulations = st.slider("Nombre de Simulations", 10, 500, 100, 10, help="Nombre d'utilisateurs simulés")
        ctr_improvement = st.slider("Amélioration CTR (%)", 0, 50, 20, 5, help="Amélioration simulée pour le groupe B")
        
        if st.button("🎲 Générer Données", key="generate_data", type="primary"):
            try:
                with st.spinner("Génération des données simulées..."):
                    np.random.seed(42)
                    
                    success_count = 0
                    for i in range(num_simulations):
                        try:
                            user_id = f"user_sim_{i}"
                            group = ab_framework.assign_user_to_group(user_id, selected_test)
                            
                            # Simuler CTR avec amélioration pour groupe B
                            if group == TestGroup.CONTROL:
                                ctr = np.random.beta(2, 8)  # ~20% CTR baseline
                            else:
                                ctr = np.random.beta(2 + ctr_improvement/10, 8)  # Amélioration
                            
                            ab_framework.track_metric(user_id, selected_test, "ctr", ctr)
                            
                            # Simuler conversion rate
                            conversion = 1 if np.random.random() < ctr else 0
                            ab_framework.track_metric(user_id, selected_test, "conversion_rate", conversion)
                            success_count += 1
                        except Exception as e:
                            st.error(f"Erreur simulation {i}: {e}")
                    
                    st.success(f"✅ {success_count}/{num_simulations} simulations générées")
                    st.session_state.data_generated[selected_test] = True
            except Exception as e:
                st.error(f"Erreur lors de la génération: {e}")
    
    with col2:
        st.markdown("""
        <div class="kpi-card">
            <h4>ℹ️ Processus de Simulation</h4>
            <ul style="font-size: 0.9rem; margin-top: 1rem;">
                <li>Assignment aléatoire aux groupes</li>
                <li>Simulation CTR avec Beta distribution</li>
                <li>Calcul conversion rate</li>
                <li>Tracking des métriques</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Analyse des résultats
    st.markdown('<h2 class="section-header">📈 Analyse des Résultats</h2>', unsafe_allow_html=True)
    
    # Vérifier si on a assez de données
    has_data = False
    for metric_name in config.metrics:
        if metric_name in test_data["metrics"]:
            metric_data = test_data["metrics"][metric_name]
            if metric_data is not None and isinstance(metric_data, dict):
                group_a_count = len(metric_data.get("A", []))
                group_b_count = len(metric_data.get("B", []))
                
                if group_a_count >= config.min_sample_size and group_b_count >= config.min_sample_size:
                    has_data = True
                    break
    
    if has_data:
        # Analyser les résultats
        results = ab_framework.analyze_test_results(selected_test)
        
        # Afficher le statut global
        if results["overall_status"] == "significant_results":
            st.markdown('<div class="success-box">✅ Résultats Statistiquement Significatifs - Le groupe B montre une amélioration significative</div>', unsafe_allow_html=True)
        elif results["overall_status"] == "no_significant_results":
            st.markdown('<div class="warning-box">⚠️ Pas de Résultats Significatifs - Différence non statistiquement significative</div>', unsafe_allow_html=True)
        else:
            st.info(f"📊 Status: {results['overall_status']}")
        
        # Visualisations par métrique
        for metric_name, analysis in results["metrics_analysis"].items():
            if isinstance(analysis, dict) and analysis.get("sample_size_sufficient", False):
                st.markdown(f"### 🎯 Métrique: {metric_name.upper()}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if "mean_a" in analysis:
                        st.markdown(f"""
                        <div class="kpi-card">
                            <p class="kpi-value">{analysis['mean_a']:.4f}</p>
                            <p class="kpi-label">Moyenne A (Control)</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    if "mean_b" in analysis:
                        st.markdown(f"""
                        <div class="kpi-card">
                            <p class="kpi-value">{analysis['mean_b']:.4f}</p>
                            <p class="kpi-label">Moyenne B (Treatment)</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col3:
                    if "improvement_percent" in analysis:
                        trend_class = "trend-positive" if analysis['improvement_percent'] > 0 else "trend-negative"
                        st.markdown(f"""
                        <div class="kpi-card">
                            <p class="kpi-value">{analysis['improvement_percent']:.2f}%</p>
                            <p class="kpi-label">Amélioration</p>
                            <p class="kpi-trend {trend_class}">📈 Performance</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Test statistique
                if "p_value" in analysis:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="kpi-card">
                            <p class="kpi-value">{analysis['p_value']:.4f}</p>
                            <p class="kpi-label">P-value</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if isinstance(analysis, dict):
                            is_significant = analysis.get("is_significant", False)
                            st.markdown(f"""
                            <div class="kpi-card">
                                <p class="kpi-value">{'✅ Oui' if is_significant else '❌ Non'}</p>
                                <p class="kpi-label">Significatif</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Visualisation des distributions
                metric_data = test_data["metrics"][metric_name]
                if metric_data is not None and isinstance(metric_data, dict):
                    group_a_values = [item["value"] for item in metric_data.get("A", [])]
                    group_b_values = [item["value"] for item in metric_data.get("B", [])]
                else:
                    group_a_values = []
                    group_b_values = []
                
                fig = go.Figure()
                
                fig.add_trace(go.Histogram(
                    x=group_a_values,
                    name='Group A (Control)',
                    opacity=0.7,
                    marker_color='#1e3a8a'
                ))
                
                fig.add_trace(go.Histogram(
                    x=group_b_values,
                    name='Group B (Treatment)',
                    opacity=0.7,
                    marker_color='#f59e0b'
                ))
                
                fig.update_layout(
                    title="",
                    xaxis_title=metric_name,
                    yaxis_title="Count",
                    barmode='overlay',
                    template='plotly_white',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Box plot
                fig_box = go.Figure()
                
                fig_box.add_trace(go.Box(
                    y=group_a_values,
                    name='Group A (Control)',
                    marker_color='#1e3a8a'
                ))
                
                fig_box.add_trace(go.Box(
                    y=group_b_values,
                    name='Group B (Treatment)',
                    marker_color='#f59e0b'
                ))
                
                fig_box.update_layout(
                    title="",
                    yaxis_title=metric_name,
                    template='plotly_white',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                
                st.plotly_chart(fig_box, use_container_width=True)
                
                # Confidence intervals
                if "ci_a" in analysis and "ci_b" in analysis:
                    fig_ci = go.Figure()
                    
                    fig_ci.add_trace(go.Bar(
                        x=['Group A', 'Group B'],
                        y=[analysis['mean_a'], analysis['mean_b']],
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            array=[
                                analysis['ci_a'][1] - analysis['mean_a'],
                                analysis['ci_b'][1] - analysis['mean_b']
                            ],
                            arrayminus=[
                                analysis['mean_a'] - analysis['ci_a'][0],
                                analysis['mean_b'] - analysis['ci_b'][0]
                            ]
                        ),
                        name='Mean with 95% CI',
                        marker_color=['#1e3a8a', '#f59e0b']
                    ))
                    
                    fig_ci.update_layout(
                        title="Intervalles de Confiance (95%)",
                        yaxis_title=metric_name,
                        template='plotly_white',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=0, r=0, t=0, b=0)
                    )
                    
                    st.plotly_chart(fig_ci, use_container_width=True)
    
    else:
        st.warning("⚠️ Données insuffisantes pour l'analyse statistique")
        st.info(f"Sample size requis: {config.min_sample_size} par groupe")
        
        # Afficher le sample size actuel
        for metric_name in config.metrics:
            if metric_name in test_data["metrics"]:
                metric_data = test_data["metrics"][metric_name]
                if metric_data is not None and isinstance(metric_data, dict):
                    group_a_count = len(metric_data.get("A", []))
                    group_b_count = len(metric_data.get("B", []))
                    
                    st.write(f"**{metric_name}**:")
                    st.write(f"  Group A: {group_a_count} / {config.min_sample_size}")
                    st.write(f"  Group B: {group_b_count} / {config.min_sample_size}")
    
    st.markdown("---")
    
    # Boutons d'action
    st.markdown('<h2 class="section-header">🎯 Actions</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Analyser Résultats (JSON)", help="Afficher les résultats bruts en JSON"):
            results = ab_framework.analyze_test_results(selected_test)
            st.json(results)
    
    with col2:
        if st.button("🛑 Arrêter Test", help="Arrêter le test actif"):
            results = ab_framework.stop_test(selected_test)
            st.success("✅ Test arrêté")
            st.json(results)
    
    with col3:
        if st.button("📋 Résumé Complet", help="Afficher le résumé complet du test"):
            summary = ab_framework.get_test_summary(selected_test)
            st.json(summary)

else:
    st.info("👋 Sélectionnez ou créez un test A/B pour commencer")

# Footer professionnel
st.markdown("""
<div class="footer">
    <p>© 2024 A/B Testing Dashboard | Système d'Expérimentation Statistique</p>
    <p style="margin-top: 0.5rem; opacity: 0.7;">Tests statistiques: T-test, Mann-Whitney U, Chi-square</p>
</div>
""", unsafe_allow_html=True)
