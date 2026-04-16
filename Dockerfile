# Dockerfile pour Moteur de Recommandation E-commerce
# Base image Python 3.13 slim
FROM python:3.13-slim

# Labels pour métadonnées
LABEL maintainer="E-commerce Recommendation Team"
LABEL description="Moteur de recommandation e-commerce avec ML et API"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Répertoire de travail
WORKDIR /app

# Copier les fichiers de configuration
COPY requirements.txt .
COPY configs/ configs/

# Installer les dépendances Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install gunicorn uvicorn[standard]

# Copier le code source
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY docs/ docs/

# Créer les répertoires nécessaires
RUN mkdir -p data/processed data/raw models logs

# Permissions
RUN chmod -R 755 /app

# Exposer le port de l'API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Commande de démarrage
CMD ["uvicorn", "src.serving.recommendation_api:app", "--host", "0.0.0.0", "--port", "8000"]
