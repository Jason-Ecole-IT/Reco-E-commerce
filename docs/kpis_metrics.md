# KPIs métier et métriques techniques

## Objectifs métier

- Augmenter le taux de clics (CTR) de 15% par rapport à un baseline aléatoire
- Améliorer le taux de conversion
- Accroître le revenu moyen par utilisateur
- Favoriser l’engagement et la diversité des recommandations

## KPIs principaux

- CTR (Click-Through Rate)
- Conversion Rate
- Revenue per User
- Diversity Score
- Coverage des recommandations

## Métriques techniques

### Performance API

- Latence cible : <50ms au 95ᵉ centile
- Throughput : >10 000 requêtes/s
- Disponibilité : >99,9%
- Taux d’erreur : <0,1%

### Performance des modèles

- RMSE : <1,0 sur jeu de test
- Precision@10 : >0,2
- NDCG@10 : >0,3
- Coverage utilisateur/item : >80%

### Infrastructure

- Cache hit rate : >85%
- Durée des jobs Spark : <30 min
- Utilisation mémoire : <80%
- I/O disque : <70%

## Qualité des données

- Freshness : mise à jour quotidienne
- Complétude : >95%
- Précision des données : >98%
- Consistance des features : >99%
- Validité des valeurs : >95%

## Métriques A/B Testing

- Traffic split : 50/50
- Durée minimale : 2 semaines
- Power statistique : 80%
- Niveau de signification : 5%
- Critère de succès principal : CTR +15%

## Monitoring

### Tableau de bord temps réel

- Temps de réponse API
- Taux de requêtes
- Taux d’erreur
- Cache hit rate
- Précision des recommandations
- Score de dérive du modèle

### Rapports quotidiens

- Volume de données traitées
- Erreurs de validation
- Valeurs manquantes
- Duplications détectées
- Durée d’entraînement

### Alertes

- Critiques : API indisponible >5 min, échec pipeline >30 min, drop modèle >20%
- Avertissements : latence >100ms, dérive du modèle, qualité donnée dégradée
- Informations : batch terminé, entraînement achevé, mise à jour de features

## Reporting

- Quotidien : synthèse performance et qualité
- Hebdomadaire : résultats A/B, tendances modèles, analyse utilisateur
- Mensuel : ROI, scalabilité et recommandations stratégiques

## Seuils de réussite

- Minimum : CTR +10%, latence <100ms, RMSE <1,2, uptime >99%
- Cible : CTR +15%, latence <50ms, RMSE <1,0, uptime >99,9%
- Excellence : CTR +25%, latence <30ms, RMSE <0,8, uptime >99,95%
