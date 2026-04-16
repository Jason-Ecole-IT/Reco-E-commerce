"""
Dashboard A/B Testing avec Streamlit
Visualisation des résultats des tests A/B
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
    page_title="Dashboard A/B Testing",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #667eea;
        font-weight: bold;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-header">🧪 Dashboard A/B Testing</h1>', unsafe_allow_html=True)

# Initialiser le framework A/B testing
if 'ab_framework' not in st.session_state:
    st.session_state.ab_framework = ABTestingFramework()
    st.session_state.ab_framework.active_tests = {}

ab_framework = st.session_state.ab_framework

# Sidebar
st.sidebar.title("⚙️ Configuration")

# Créer un nouveau test
with st.sidebar.expander("➕ Créer Nouveau Test", expanded=False):
    test_name = st.text_input("Nom du Test", "Recommendation CTR Test")
    description = st.text_area("Description", "Test CTR des recommandations ML vs Random")
    traffic_split = st.slider("Split Traffic (A/B)", 0.1, 0.9, 0.5, 0.1)
    min_sample_size = st.number_input("Sample Size Min", 100, 10000, 1000, 100)
    
    metrics = st.multiselect(
        "Métriques à tracker",
        ["ctr", "conversion_rate", "avg_session_duration", "revenue_per_user"],
        default=["ctr", "conversion_rate"]
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

# Liste des tests actifs
st.sidebar.title("📋 Tests Actifs")

if ab_framework.active_tests:
    test_ids = list(ab_framework.active_tests.keys())
    selected_test = st.sidebar.selectbox("Sélectionner un Test", test_ids)
    
    if selected_test:
        test_data = ab_framework.active_tests[selected_test]
        config = test_data["config"]
        
        st.sidebar.info(f"""
        **Test**: {config.test_name}
        **Status**: {test_data["status"]}
        **Split**: {config.traffic_split * 100}% / {(1 - config.traffic_split) * 100}%
        **Sample Size**: {config.min_sample_size}
        **Assignments**: {len(test_data["assignments"])}
        """)
else:
    st.sidebar.warning("Aucun test actif")
    selected_test = None

# Section principale
if selected_test:
    test_data = ab_framework.active_tests[selected_test]
    config = test_data["config"]
    
    # En-tête du test
    st.header(f"📊 Test: {config.test_name}")
    st.markdown(f"*{config.description}*")
    
    # Statistiques du test
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Status", test_data["status"])
    
    with col2:
        st.metric("Total Assignments", len(test_data["assignments"]))
    
    with col3:
        group_a_count = sum(1 for g in test_data["assignments"].values() if g == TestGroup.CONTROL)
        group_b_count = sum(1 for g in test_data["assignments"].values() if g == TestGroup.TREATMENT)
        st.metric("Group A (Control)", group_a_count)
    
    with col4:
        st.metric("Group B (Treatment)", group_b_count)
    
    # Simulation de données pour démonstration
    st.subheader("🎮 Simulation de Données")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        num_simulations = st.slider("Nombre de Simulations", 10, 500, 100, 10)
        ctr_improvement = st.slider("Amélioration CTR (%)", 0, 50, 20, 5)
        
        if st.button("🎲 Générer Données", type="primary"):
            np.random.seed(42)
            
            for i in range(num_simulations):
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
            
            st.success(f"✅ {num_simulations} simulations générées")
            st.rerun()
    
    with col2:
        st.info("""
        **Instructions:**
        
        1. Créez un test dans la sidebar
        2. Sélectionnez le test
        3. Générez des données simulées
        4. Analysez les résultats
        """)
    
    # Analyse des résultats
    st.subheader("📈 Analyse des Résultats")
    
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
            st.markdown('<div class="success-box">✅ Résultats Statistiquement Significatifs</div>', unsafe_allow_html=True)
        elif results["overall_status"] == "no_significant_results":
            st.markdown('<div class="warning-box">⚠️ Pas de Résultats Significatifs</div>', unsafe_allow_html=True)
        else:
            st.info(f"📊 Status: {results['overall_status']}")
        
        # Visualisations par métrique
        for metric_name, analysis in results["metrics_analysis"].items():
            if analysis.get("sample_size_sufficient", False):
                st.subheader(f"🎯 Métrique: {metric_name.upper()}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if "mean_a" in analysis:
                        st.metric("Moyenne A", f"{analysis['mean_a']:.4f}")
                
                with col2:
                    if "mean_b" in analysis:
                        st.metric("Moyenne B", f"{analysis['mean_b']:.4f}")
                
                with col3:
                    if "improvement_percent" in analysis:
                        st.metric("Amélioration", f"{analysis['improvement_percent']:.2f}%")
                
                # Test statistique
                if "p_value" in analysis:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("P-value", f"{analysis['p_value']:.4f}")
                    
                    with col2:
                        is_significant = analysis.get("is_significant", False)
                        st.metric("Significatif", "✅ Oui" if is_significant else "❌ Non")
                
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
                    marker_color='blue'
                ))
                
                fig.add_trace(go.Histogram(
                    x=group_b_values,
                    name='Group B (Treatment)',
                    opacity=0.7,
                    marker_color='orange'
                ))
                
                fig.update_layout(
                    title=f"Distribution {metric_name}",
                    xaxis_title=metric_name,
                    yaxis_title="Count",
                    barmode='overlay'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Box plot
                fig_box = go.Figure()
                
                fig_box.add_trace(go.Box(
                    y=group_a_values,
                    name='Group A (Control)',
                    marker_color='blue'
                ))
                
                fig_box.add_trace(go.Box(
                    y=group_b_values,
                    name='Group B (Treatment)',
                    marker_color='orange'
                ))
                
                fig_box.update_layout(
                    title=f"Box Plot {metric_name}",
                    yaxis_title=metric_name
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
                        marker_color=['blue', 'orange']
                    ))
                    
                    fig_ci.update_layout(
                        title=f"Intervalles de Confiance (95%) - {metric_name}",
                        yaxis_title=metric_name
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
    
    # Boutons d'action
    st.subheader("🎯 Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Analyser Résultats"):
            results = ab_framework.analyze_test_results(selected_test)
            st.json(results)
    
    with col2:
        if st.button("🛑 Arrêter Test"):
            results = ab_framework.stop_test(selected_test)
            st.success("✅ Test arrêté")
            st.json(results)
    
    with col3:
        if st.button("📋 Résumé Complet"):
            summary = ab_framework.get_test_summary(selected_test)
            st.json(summary)

else:
    st.info("👋 Sélectionnez ou créez un test A/B pour commencer")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🧪 Dashboard A/B Testing - Powered by Streamlit</p>
    <p>Tests statistiques: T-test, Mann-Whitney U, Chi-square</p>
</div>
""", unsafe_allow_html=True)
