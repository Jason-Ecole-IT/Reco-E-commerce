# Architecture Diagrams

Cette page contient les diagrammes d'architecture du projet. Ces diagrammes développe l'intégralité du process, du moteur de recommandation ainsi que l'interface utilisateur.

## Diagramme process client

````mermaid
graph LR

    subgraph "Cold Start"
        A[New User]
        B[Collect User Information]
        C[Generate Metadata]
        D[Generate Recommendations]
        E[Display Recommendations]
        A --> B --> C --> D --> E
    end

    subgraph "Existing User"
        F[Existing User]
        G[Generate Recommendations]
        H[Display Recommendations]
        F --> G --> H
    end
````
