---
title: First Agent Template
emoji: "🤖"
colorFrom: pink
colorTo: yellow
sdk: gradio
sdk_version: 5.15.0
app_file: app.py
pinned: false
tags:
- smolagents
- agent
- smolagent
- tool
- agent-course
---

# Simple Local Agent

Un agent conversationnel simple utilisant SmoLAgents pour se connecter à un modèle de langage, que ce soit via un serveur local (LMStudio) ou via d'autres APIs.

## Prérequis

- Python 3.8+
- Un modèle de langage hébergé localement ou accessible via une API

## Installation

1. Installez les dépendances requises :

```bash
pip install -r requirements.txt
```

## Utilisation

### Interface Gradio

1. Assurez-vous que votre serveur LLM est en cours d'exécution à l'adresse spécifiée.

2. Lancez l'agent avec l'interface Gradio :

```bash
python app.py
```

### Interface Streamlit (Nouvelle !)

Nous avons également ajouté une interface Streamlit qui offre plus de flexibilité et d'options de configuration :

1. Lancez l'application Streamlit :

```bash
streamlit run streamlit_app.py
```

2. Accédez à l'interface via votre navigateur web (généralement à l'adresse http://localhost:8501).

### Fonctionnalités de l'interface Streamlit

- **Interface de chat interactive** pour discuter avec l'agent
- **Choix entre différents types de modèles** :
  - OpenAI Server (LMStudio ou autre serveur compatible OpenAI)
  - Hugging Face API
  - Hugging Face Cloud
- **Configuration personnalisable** pour chaque type de modèle
- **Affichage en temps réel** du raisonnement de l'agent
- **Informations utiles** dans la barre latérale

## Configuration ⚙️

### Configuration du modèle

L'interface Streamlit permet de configurer facilement le modèle sans modifier le code source :

- **OpenAI Server** : URL du serveur, ID du modèle, clé API
- **Hugging Face API** : URL du modèle, tokens maximum, température
- **Hugging Face Cloud** : URL de l'endpoint, tokens maximum, température

### Configuration des outils

L'agent est équipé de plusieurs outils puissants qui lui permettent d'interagir avec le monde extérieur et d'effectuer diverses actions :

#### Outils principaux intégrés

- **DuckDuckGoSearchTool** : Permet à l'agent d'effectuer des recherches web via DuckDuckGo pour obtenir des informations à jour sur n'importe quel sujet.
- **VisitWebpageTool** : Permet à l'agent de visiter une page web spécifique et d'en extraire le contenu pour analyse.
- **ShellCommandTool** : Donne à l'agent la capacité d'exécuter des commandes shell sur le système hôte (avec les précautions de sécurité appropriées).
- **CreateFileTool** : Permet à l'agent de créer de nouveaux fichiers dans le système.
- **ModifyFileTool** : Permet à l'agent de modifier des fichiers existants.
- **FinalAnswerTool** : Fournit une réponse finale structurée à l'utilisateur, résumant les informations trouvées.

#### Outils personnalisés

L'agent inclut également quelques outils personnalisés :

- **get_current_realtime** : Renvoie l'heure actuelle du système.
- **get_current_time_in_timezone** : Récupère l'heure locale actuelle dans un fuseau horaire spécifié (par exemple, "Europe/Paris", "America/New_York").

#### Extensibilité

L'architecture de l'agent est conçue pour être facilement extensible. Vous pouvez ajouter vos propres outils personnalisés en suivant le modèle d'exemple dans le fichier `app.py` :

```python
@tool
def my_custom_tool(arg1: str, arg2: int) -> str:
    """Description de ce que fait l'outil
    Args:
        arg1: description du premier argument
        arg2: description du second argument
    """
    # Implémentation de votre outil
    return "Résultat de l'outil"
```

## Exemples d'utilisation

Voici quelques exemples de questions que vous pouvez poser à l'agent :

- "Quelle est l'heure actuelle à Tokyo ?"
- "Peux-tu me faire un résumé des dernières nouvelles sur l'IA ?"
- "Crée un fichier contenant un exemple de code Python pour trier une liste"
- "Explique-moi comment fonctionne la technologie des transformers en IA"

---

*Consultez la référence de configuration sur https://huggingface.co/docs/hub/spaces-config-reference*
