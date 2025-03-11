import streamlit as st
import os
import sys
import yaml
import datetime
import pytz
from typing import List, Dict, Any

# Ajout du répertoire courant au chemin Python pour importer les modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import des composants nécessaires pour l'agent
from smolagents import CodeAgent
from smolagents.models import OpenAIServerModel, HfApiModel
from tools.final_answer import FinalAnswerTool
from tools.visit_webpage import VisitWebpageTool
from tools.web_search import DuckDuckGoSearchTool
from tools.shell_tool import ShellCommandTool
from tools.create_file_tool import CreateFileTool
from tools.modify_file_tool import ModifyFileTool

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Agent Conversationnel SmoLAgents 🤖",
    page_icon="🤖",
    layout="wide",
)

def initialize_agent(model_type="openai_server", model_config=None):
    """Initialise l'agent avec les outils et le modèle choisi
    
    Args:
        model_type: Type de modèle à utiliser ('openai_server', 'hf_api', etc.)
        model_config: Configuration spécifique au modèle
    """
    
    # Configuration du modèle en fonction du type choisi
    if model_type == "openai_server":
        # Configuration par défaut pour OpenAIServerModel
        if model_config is None:
            model_config = {
                "api_base": "http://192.168.1.141:1234/v1",
                "model_id": "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
                "api_key": "sk-dummy-key"
            }
        
        model = OpenAIServerModel(
            api_base=model_config["api_base"],
            model_id=model_config["model_id"],
            api_key=model_config["api_key"]
        )
    
    elif model_type == "hf_api":
        # Configuration par défaut pour HfApiModel
        if model_config is None:
            model_config = {
                "model_id": "http://192.168.1.141:1234/v1",
                "max_new_tokens": 2096,
                "temperature": 0.5
            }
        
        model = HfApiModel(
            model_id=model_config["model_id"],
            max_new_tokens=model_config["max_new_tokens"],
            temperature=model_config["temperature"]
        )
    
    elif model_type == "hf_cloud":
        # Configuration pour HfApiModel avec un endpoint cloud
        if model_config is None:
            model_config = {
                "model_id": "https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud",
                "max_new_tokens": 2096,
                "temperature": 0.5
            }
        
        model = HfApiModel(
            model_id=model_config["model_id"],
            max_new_tokens=model_config["max_new_tokens"],
            temperature=model_config["temperature"]
        )
    
    else:
        st.error(f"Type de modèle non supporté: {model_type}")
        return None
    
    # Chargement des templates de prompt depuis le fichier YAML
    try:
        with open("prompts.yaml", 'r') as stream:
            prompt_templates = yaml.safe_load(stream)
    except:
        st.error("Impossible de charger prompts.yaml. Utilisation des prompts par défaut.")
        prompt_templates = None
    
    # Initialisation des outils
    final_answer = FinalAnswerTool()
    
    # Création de l'agent avec les mêmes outils que dans app.py
    agent = CodeAgent(
        model=model,
        tools=[
            final_answer, 
            DuckDuckGoSearchTool(), 
            VisitWebpageTool(), 
            ShellCommandTool(), 
            CreateFileTool(), 
            ModifyFileTool()
        ],
        max_steps=6,
        verbosity_level=1,
        grammar=None,
        planning_interval=None,
        name=None,
        description=None,
        prompt_templates=prompt_templates
    )
    
    return agent

def format_step_message(step, is_final=False):
    """Formate les messages de l'agent pour l'affichage dans Streamlit"""
    
    if hasattr(step, "model_output") and step.model_output:
        # Nettoyer et formater la sortie du modèle pour l'affichage
        content = step.model_output.strip()
        if not is_final:
            return content
        else:
            return f"**Réponse finale :** {content}"
    
    if hasattr(step, "observations") and step.observations:
        # Afficher les observations des outils
        return f"**Observations :** {step.observations.strip()}"
    
    if hasattr(step, "error") and step.error:
        # Afficher les erreurs
        return f"**Erreur :** {step.error}"
    
    # Cas par défaut
    return str(step)

def process_user_input(agent, user_input):
    """Traite l'entrée utilisateur avec l'agent et renvoie les résultats étape par étape"""
    
    # Vérification de la connexion au serveur LLM
    try:
        # Exécution de l'agent et capture des étapes
        with st.spinner("L'agent réfléchit..."):
            # Placeholder pour la sortie de l'agent
            response_container = st.container()
            
            # Exécution de l'agent et capture des étapes
            steps = []
            final_step = None
            
            with response_container:
                step_container = st.empty()
                step_text = ""
                
                # Exécute l'agent et capture les étapes de manière incrémentale
                for step in agent.run(user_input, stream=True):
                    steps.append(step)
                    
                    # Mettre à jour l'affichage des étapes
                    step_number = f"Étape {step.step_number}" if hasattr(step, "step_number") and step.step_number is not None else ""
                    step_content = format_step_message(step)
                    
                    # Ajouter au texte des étapes
                    if step_number:
                        step_text += f"### {step_number}\n\n"
                    step_text += f"{step_content}\n\n---\n\n"
                    
                    # Mettre à jour l'affichage
                    step_container.markdown(step_text)
                    
                    # Conserver la dernière étape pour la réponse finale
                    final_step = step
                
                # Afficher la réponse finale
                if final_step:
                    final_answer = format_step_message(final_step, is_final=True)
                    st.markdown(f"## Réponse Finale\n\n{final_answer}")
            
            return final_step
    except Exception as e:
        st.error(f"Erreur lors de l'exécution de l'agent: {str(e)}")
        return None

def main():
    st.title("Agent Conversationnel SmoLAgents 🤖")
    
    st.markdown("""
    Bienvenue! Cet agent utilise SmoLAgents pour se connecter à un modèle de langage.
    Posez vos questions ci-dessous.
    """)
    
    # Sidebar pour la configuration du modèle
    with st.sidebar:
        st.title("Configuration du Modèle")
        
        # Sélectionner le type de modèle
        model_type = st.selectbox(
            "Type de modèle",
            ["openai_server", "hf_api", "hf_cloud"],
            index=0,
            help="Choisissez le type de modèle à utiliser avec l'agent"
        )
        
        # Configuration spécifique en fonction du type de modèle
        model_config = {}
        
        if model_type == "openai_server":
            st.subheader("Configuration OpenAI Server")
            model_config["api_base"] = st.text_input(
                "URL du serveur",
                value="http://192.168.1.141:1234/v1",
                help="Adresse du serveur OpenAI compatible"
            )
            model_config["model_id"] = st.text_input(
                "ID du modèle",
                value="Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
                help="Identifiant du modèle local"
            )
            model_config["api_key"] = st.text_input(
                "Clé API",
                value="sk-dummy-key",
                type="password",
                help="Clé API pour le serveur (dummy pour LMStudio)"
            )
        
        elif model_type == "hf_api":
            st.subheader("Configuration Hugging Face API")
            model_config["model_id"] = st.text_input(
                "URL du modèle",
                value="http://192.168.1.141:1234/v1",
                help="URL du modèle ou endpoint"
            )
            model_config["max_new_tokens"] = st.slider(
                "Tokens maximum",
                min_value=512,
                max_value=4096,
                value=2096,
                help="Nombre maximum de tokens à générer"
            )
            model_config["temperature"] = st.slider(
                "Température",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Température pour la génération (plus élevée = plus créatif)"
            )
        
        elif model_type == "hf_cloud":
            st.subheader("Configuration Hugging Face Cloud")
            model_config["model_id"] = st.text_input(
                "URL du endpoint cloud",
                value="https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud",
                help="URL de l'endpoint cloud Hugging Face"
            )
            model_config["max_new_tokens"] = st.slider(
                "Tokens maximum",
                min_value=512,
                max_value=4096,
                value=2096,
                help="Nombre maximum de tokens à générer"
            )
            model_config["temperature"] = st.slider(
                "Température",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Température pour la génération (plus élevée = plus créatif)"
            )
        
        # Bouton pour réinitialiser l'agent avec la nouvelle configuration
        if st.button("Appliquer la configuration"):
            with st.spinner("Initialisation de l'agent avec le nouveau modèle..."):
                st.session_state.agent = initialize_agent(model_type, model_config)
                st.success("✅ Configuration appliquée avec succès!")
    
    # Vérifier la connexion au serveur
    if model_type == "openai_server":
        llm_api_url = model_config["api_base"].split("/v1")[0]
        try:
            import requests
            response = requests.get(f"{llm_api_url}/health", timeout=2)
            if response.status_code == 200:
                st.success("✅ Connexion au serveur LLM établie")
            else:
                st.warning("⚠️ Le serveur LLM est accessible mais renvoie un statut non-OK")
        except Exception:
            st.error("❌ Impossible de se connecter au serveur LLM. Vérifiez que le serveur est en cours d'exécution à l'adresse spécifiée.")
    
    # Initialisation de l'agent si ce n'est pas déjà fait
    if "agent" not in st.session_state:
        with st.spinner("Initialisation de l'agent..."):
            st.session_state.agent = initialize_agent(model_type, model_config)
    
    # Initialisation de l'historique de conversation
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Bonjour! Comment puis-je vous aider aujourd'hui?"}
        ]
    
    # Affichage de l'historique des messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Zone de saisie utilisateur
    if prompt := st.chat_input("Posez votre question..."):
        # Ajouter la question de l'utilisateur à l'historique
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Afficher la question de l'utilisateur
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Traiter la demande avec l'agent
        with st.chat_message("assistant"):
            response = process_user_input(st.session_state.agent, prompt)
            
            if response and hasattr(response, "model_output"):
                # Ajouter la réponse à l'historique
                st.session_state.messages.append({"role": "assistant", "content": response.model_output})
    
    # Bouton pour effacer l'historique
    if st.sidebar.button("Nouvelle conversation"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Bonjour! Comment puis-je vous aider aujourd'hui?"}
        ]
        st.rerun()
    
    # Afficher des informations supplémentaires dans la barre latérale
    with st.sidebar:
        st.title("À propos de cet agent")
        st.markdown("""
        Cet agent utilise SmoLAgents pour se connecter à un modèle de langage hébergé localement.
        
        ### Outils disponibles
        - Recherche web (DuckDuckGo)
        - Visite de pages web
        - Exécution de commandes shell
        - Création et modification de fichiers
        
        ### Configuration
        Utilisez les options ci-dessus pour configurer le modèle de langage.
        
        ### Problèmes courants
        - Si l'agent ne répond pas, vérifiez que le serveur LLM est en cours d'exécution et accessible.
        - Assurez-vous que toutes les dépendances sont installées via `pip install -r requirements.txt`.
        """)
        
        # Afficher l'heure actuelle dans différents fuseaux horaires
        st.subheader("Heure actuelle")
        selected_timezone = st.selectbox(
            "Choisissez un fuseau horaire",
            ["Europe/Paris", "America/New_York", "Asia/Tokyo", "Australia/Sydney"]
        )
        
        tz = pytz.timezone(selected_timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        st.write(f"L'heure actuelle à {selected_timezone} est: {local_time}")

if __name__ == "__main__":
    main() 