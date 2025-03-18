# =============================================================================
# STREAMLIT APPLICATION FOR SMOLAGENTS CONVERSATIONAL AGENT
# =============================================================================
# This application provides a web interface for interacting with a SmoLAgents-based
# conversational agent. It supports multiple model backends, visualization capabilities,
# and a rich chat interface.
# =============================================================================

# Standard library imports
import streamlit as st
import os
import sys
import yaml
import datetime
import pytz
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple

# Add current directory to Python path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# SmoLAgents and related imports
from smolagents import CodeAgent
from smolagents.models import OpenAIServerModel, HfApiModel
from smolagents.memory import ToolCall

# Tool imports for agent capabilities
from tools.final_answer import FinalAnswerTool
from tools.validate_final_answer import ValidateFinalAnswer
from tools.visit_webpage import VisitWebpageTool
from tools.web_search import DuckDuckGoSearchTool
from tools.shell_tool import ShellCommandTool
from tools.create_file_tool import CreateFileTool
from tools.modify_file_tool import ModifyFileTool

# # Telemetry imports (currently disabled)
# from phoenix.otel import register
# from openinference.instrumentation.smolagents import SmolagentsInstrumentor
# register()
# SmolagentsInstrumentor().instrument()

# Visualization functionality imports
from visualizations import (
    create_line_chart,
    create_bar_chart,
    create_scatter_plot,
    detect_visualization_request,
    generate_sample_data
)

# Configure Streamlit page settings
st.set_page_config(
    page_title="Streamlit generator 🤖",
    page_icon="🤖",
    layout="wide",  # Use wide layout for better display of content
)

def initialize_agent(model_type="openai_server", model_config=None):
    """Initialize the agent with the specified model and tools.
    
    This function creates a SmoLAgents CodeAgent instance with the specified language model
    and a set of tools that enable various capabilities like web search, file operations,
    and shell command execution.
    
    Args:
        model_type (str): Type of model to use. Options are:
            - 'openai_server': For OpenAI-compatible API servers (like LMStudio or OpenRouter)
            - 'hf_api': For Hugging Face API endpoints
            - 'hf_cloud': For Hugging Face cloud endpoints
        model_config (dict, optional): Configuration dictionary for the model.
            If None, default configurations will be used.
    
    Returns:
        CodeAgent: Initialized agent instance, or None if model type is not supported
    """
    
    # Configure the model based on the selected type
    if model_type == "openai_server":
                
        # Initialize OpenAI-compatible model
        model = OpenAIServerModel(
            api_base=model_config["api_base"],
            model_id=model_config["model_id"],
            api_key=model_config["api_key"],
            max_tokens=12000  # Maximum tokens for response generation
        )
    
    elif model_type == "hf_api":
        # Default configuration for local Hugging Face API endpoint
        if model_config is None:
            model_config = {
                "model_id": "http://192.168.1.141:1234/v1",  # Local API endpoint
                "max_new_tokens": 2096,
                "temperature": 0.5  # Controls randomness (0.0 = deterministic, 1.0 = creative)
            }
        
        # Initialize Hugging Face API model
        model = HfApiModel(
            model_id=model_config["model_id"],
            max_new_tokens=model_config["max_new_tokens"],
            temperature=model_config["temperature"]
        )
    
    elif model_type == "hf_cloud":
        # Default configuration for Hugging Face cloud endpoint
        if model_config is None:
            model_config = {
                "model_id": "https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud",
                "max_new_tokens": 2096,
                "temperature": 0.5
            }
        
        # Initialize Hugging Face cloud model
        model = HfApiModel(
            model_id=model_config["model_id"],
            max_new_tokens=model_config["max_new_tokens"],
            temperature=model_config["temperature"]
        )
    
    else:
        # Handle unsupported model types
        st.error(f"Type de modèle non supporté: {model_type}")
        return None
    
    # Load prompt templates from YAML file
    try:
        with open("prompts.yaml", 'r') as stream:
            prompt_templates = yaml.safe_load(stream)
    except:
        st.error("Impossible de charger prompts.yaml. Utilisation des prompts par défaut.")
        prompt_templates = None
    
    
    # Create the agent with tools and configuration
    agent = CodeAgent(
        model=model,
        tools=[
            # Core tools for agent functionality
            FinalAnswerTool(),          # Provides final answers to user queries
            ValidateFinalAnswer(),      # Validates final answers for quality
            DuckDuckGoSearchTool(),     # Enables web search capabilities
            VisitWebpageTool(),         # Allows visiting and extracting content from webpages
            # ShellCommandTool(),         # Enables execution of shell commands
            # CreateFileTool(),           # Allows creation of new files
            # ModifyFileTool()            # Enables modification of existing files
        ],
        max_steps=5,                   # Maximum number of reasoning steps
        verbosity_level=1,              # Level of detail in agent's output
        grammar=None,                   # Optional grammar for structured output
        planning_interval=None,         # How often to re-plan (None = no explicit planning)
        name=None,                      # Agent name
        description=None,               # Agent description
        prompt_templates=prompt_templates,  # Custom prompt templates
        # Additional Python modules the agent is allowed to import in generated code
        additional_authorized_imports=["pandas", "numpy", "matplotlib", "seaborn", "plotly", "requests", "yaml", "yfinance", "datetime", "pytz"]
    )
    
    return agent

def format_step_message(step, is_final=False):
    """Format agent messages for display in Streamlit.
    
    This function processes different types of agent step outputs (model outputs,
    observations, errors) and formats them for display in the Streamlit interface.
    
    Args:
        step: The agent step object containing output information
        is_final (bool): Whether this is the final answer step
    
    Returns:
        str: Formatted message ready for display
    """
    
    if hasattr(step, "model_output") and step.model_output:
        # Format the model's output (the agent's thinking or response)
        content = step.model_output.strip()
        if not is_final:
            return content
        else:
            # Add special formatting for final answers
            return f"**Réponse finale :** {content}"
    
    if hasattr(step, "observations") and step.observations:
        # Format tool observations (results from tool executions)
        return f"**Observations :** {step.observations.strip()}"
    
    if hasattr(step, "error") and step.error:
        # Format any errors that occurred during agent execution
        return f"**Erreur :** {step.error}"
    
    # Default case - convert step to string
    return str(step)

def process_visualization_request(user_input: str) -> Tuple[bool, Optional[st.delta_generator.DeltaGenerator]]:
    """
    Process a visualization request from the user.
    
    This function detects if the user is requesting a data visualization,
    generates appropriate sample data, and creates the requested chart.
    
    Args:
        user_input (str): The user's input message
        
    Returns:
        Tuple[bool, Optional[st.delta_generator.DeltaGenerator]]:
            - Boolean indicating if a visualization was processed
            - The Streamlit container if a visualization was created, None otherwise
    """
    # Use NLP to detect if this is a visualization request and extract details
    viz_info = detect_visualization_request(user_input)
    
    # If not a visualization request or chart type couldn't be determined, return early
    if not viz_info['is_visualization'] or not viz_info['chart_type']:
        return False, None
    
    # Extract information from the request
    chart_type = viz_info['chart_type']
    data_description = viz_info['data_description']
    parameters = viz_info['parameters']
    
    # Generate appropriate sample data based on the description and chart type
    data = generate_sample_data(data_description, chart_type)
    
    # Set default parameters if not provided by the user
    title = parameters.get('title', f"{chart_type.capitalize()} Chart" + (f" of {data_description}" if data_description else ""))
    x_label = parameters.get('x_label', data.columns[0] if len(data.columns) > 0 else "X-Axis")
    y_label = parameters.get('y_label', data.columns[1] if len(data.columns) > 1 else "Y-Axis")
    
    # Create the appropriate chart based on the requested type
    fig = None
    if chart_type == 'line':
        fig = create_line_chart(data, title=title, x_label=x_label, y_label=y_label)
    elif chart_type == 'bar':
        fig = create_bar_chart(data, title=title, x_label=x_label, y_label=y_label)
    elif chart_type == 'scatter':
        fig = create_scatter_plot(data, title=title, x_label=x_label, y_label=y_label)
    
    # If a chart was successfully created, display it
    if fig:
        # Create a container for the visualization
        viz_container = st.container()
        with viz_container:
            st.plotly_chart(fig, use_container_width=True)
        
        return True, viz_container
    
    return False, None

def process_user_input(agent, user_input):
    """Process user input with the agent and return results step by step.
    
    This function handles the execution of the agent with the user's input,
    displays the agent's thinking process in real-time, and returns the final result.
    It also handles visualization requests by integrating with the visualization system.
    
    Args:
        agent: The initialized SmoLAgents agent instance
        user_input (str): The user's query or instruction
        
    Returns:
        tuple or None: A tuple containing the final answer and a boolean flag,
                      or None if an error occurred
    """
    
    # First check if this is a visualization request
    is_viz_request, viz_container = process_visualization_request(user_input)
    
    # Even for visualization requests, we still run the agent to provide context and explanation
    
    # Execute the agent and handle any exceptions
    try:
        # Show a spinner while the agent is thinking
        with st.spinner("L'agent réfléchit..."):
            # Create a container for the agent's output
            response_container = st.container()
            
            # Initialize variables to track steps and final result
            steps = []
            final_step = None
            
            # Display the agent's thinking process in real-time
            with response_container:
                step_container = st.empty()
                step_text = ""
                
                # Execute the agent and stream results incrementally
                for step in agent.run(user_input, stream=True):
                    steps.append(step)
                    
                    # Format the current step for display
                    step_number = f"Étape {step.step_number}" if hasattr(step, "step_number") and step.step_number is not None else ""
                    step_content = format_step_message(step)
                    
                    # Build the cumulative step text
                    if step_number:
                        step_text += f"### {step_number}\n\n"
                    step_text += f"{step_content}\n\n---\n\n"
                    
                    # Update the display with the latest step information
                    # step_container.markdown(step_text)
                    
                    # Keep track of the final step for the response
                    final_step = step
                
                # Process and return the final answer
                if final_step:
                    final_answer = format_step_message(final_step, is_final=True)
                    
                    # If this was a visualization request, add a note about it
                    if is_viz_request:
                        final_answer += "\n\n*Une visualisation a été générée en fonction de votre demande.*"
                    
                    # Return the final answer with a flag indicating success
                    return (final_answer, True)
            
            # If we somehow exit the loop without a final step
            return final_step
            
    except Exception as e:
        # Handle any errors that occur during agent execution
        st.error(f"Erreur lors de l'exécution de l'agent: {str(e)}")
        return None
    
@st.fragment
def launch_app(code_to_launch):
    """Execute code within a Streamlit fragment to prevent page reloads.
    
    This function is decorated with @st.fragment to ensure that only this specific
    part of the UI is updated when code is executed, without reloading the entire page.
    This is particularly useful for executing code generated by the agent.
    
    Args:
        code_to_launch (str): Python code string to be executed
    """
    with st.container(border = True):
        app_tab, source_tab = st.tabs(["Application", "Code source"])
        with app_tab:
            # Execute the code within a bordered container for visual separation
            exec(code_to_launch)
        with source_tab:
            # Display the generated code for reference
            st.code(code_to_launch, language="python")
            st.info("Pour mettre en ligne votre application suivre le lien suivant : [Export Streamlit App](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)")

        
    return

def main():
    """Main application entry point.
    
    This function sets up the Streamlit interface, initializes the agent,
    manages the conversation history, and handles user interactions.
    It's the central orchestrator of the application's functionality.
    """
    # Set up the main page title and welcome message
    st.title("🤖 Streamlit generator")
    
    st.markdown("""
    Bienvenue! Cet agent utilise SmoLAgents pour se connecter à un modèle de langage.
    Posez vos questions ci-dessous.
    """)
    
    # Set up the sidebar for model configuration
    with st.sidebar:
        # Display the application icon
        st.title("🤖 Streamlit generator")
        # st.image("ico.webp", width=100, caption="SmoLAgents Icon")
        
        with st.expander("🛠️ Configuration du Modèle", expanded=True):  
            # Model type selection dropdown
            model_type = st.selectbox(
                "Type de modèle",
                ["Par défaut", "openai_server", "hf_api", "hf_cloud"],
                index=0,
                help="Choisissez le type de modèle à utiliser avec l'agent"
            )
            
            # Initialize empty configuration dictionary
            model_config = {}
            if model_type == "Par défaut":
                st.success("Modèle par défaut 🟢")
                
                model_config["api_base"] = "https://generativelanguage.googleapis.com/v1beta/openai/"
                model_config["model_id"] = "gemini-2.0-pro-exp-02-05"
                model_config["api_key"] = st.secrets["API_GEMINI_KEY"] #os.getenv("OPEN_ROUTER_TOKEN") or "dummy",
                model_type = "openai_server"

            # Dynamic configuration UI based on selected model type
            elif model_type == "openai_server":
                st.subheader("Configuration OpenAI Server")
                # OpenAI-compatible server URL (OpenRouter, LMStudio, etc.)
                model_config["api_base"] = st.text_input(
                    "URL du serveur",
                    value="https://openrouter.ai/api/v1",
                    help="Adresse du serveur OpenAI compatible"
                )
                # Model ID to use with the server
                model_config["model_id"] = st.text_input(
                    "ID du modèle",
                    value="google/gemini-2.0-pro-exp-02-05:free",
                    help="Identifiant du modèle local"
                )
                # API key for authentication
                model_config["api_key"] = st.text_input(
                    "Clé API",
                    value=os.getenv("OPEN_ROUTER_TOKEN") or "dummy",
                    type="password",
                    help="Clé API pour le serveur (dummy pour LMStudio)"
                )
            
            elif model_type == "hf_api":
                st.subheader("Configuration Hugging Face API")
                # Hugging Face API endpoint URL
                model_config["model_id"] = st.text_input(
                    "URL du modèle",
                    value="http://192.168.1.141:1234/v1",
                    help="URL du modèle ou endpoint"
                )
                # Maximum tokens to generate in responses
                model_config["max_new_tokens"] = st.slider(
                    "Tokens maximum",
                    min_value=512,
                    max_value=4096,
                    value=2096,
                    help="Nombre maximum de tokens à générer"
                )
                # Temperature controls randomness in generation
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
                # Hugging Face cloud endpoint URL
                model_config["model_id"] = st.text_input(
                    "URL du endpoint cloud",
                    value="https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud",
                    help="URL de l'endpoint cloud Hugging Face"
                )
                # Maximum tokens to generate in responses
                model_config["max_new_tokens"] = st.slider(
                    "Tokens maximum",
                    min_value=512,
                    max_value=4096,
                    value=2096,
                    help="Nombre maximum de tokens à générer"
                )
                # Temperature controls randomness in generation
                model_config["temperature"] = st.slider(
                    "Température",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    help="Température pour la génération (plus élevée = plus créatif)"
                )
            
            # Button to apply configuration changes and reinitialize the agent
            if st.button("Appliquer la configuration"):
                with st.spinner("Initialisation de l'agent avec le nouveau modèle..."):
                    st.session_state.agent = initialize_agent(model_type, model_config)
                    st.success("✅ Configuration appliquée avec succès!")
    
    # Check server connection for OpenAI server type
    if model_type == "openai_server":
        # Extract base URL for health check
        llm_api_url = model_config["api_base"].split("/v1")[0]
        try:
            # Attempt to connect to the server's health endpoint
            import requests
            response = requests.get(f"{llm_api_url}/health", timeout=2)
            if response:
                st.success("✅ Connexion au serveur LLM établie")
        except Exception:
            st.error("❌ Impossible de se connecter au serveur LLM. Vérifiez que le serveur est en cours d'exécution à l'adresse spécifiée.")
    
    # Initialize the agent if not already in session state
    if "agent" not in st.session_state:
        with st.spinner("Initialisation de l'agent..."):
            st.session_state.agent = initialize_agent(model_type, model_config)
    
    # Initialize conversation history if not already in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Bonjour! Comment puis-je vous aider aujourd'hui?"}
        ]
    
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input area
    if prompt := st.chat_input("Posez votre question..."):
        # Add user question to conversation history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user question
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process user input with the agent and display response
        with st.chat_message("assistant"):
            # Get response from agent
            response = process_user_input(st.session_state.agent, prompt)
            
            # If response contains executable code, run it in a fragment
            if response is not None and response[1] == True:
                launch_app(response[0])
                    
            # Add agent's response to conversation history
            if response and hasattr(response, "model_output"):
                st.session_state.messages.append({"role": "assistant", "content": response.model_output})
    
   
    # Additional information and features in the sidebar
    with st.sidebar:
        with st.container(border = True):
            st.markdown(f"🤖 Modèle sélectionné: \n\n `{model_config["model_id"]}`")
            # Button to clear conversation history and start a new chat
            if st.button("Nouvelle conversation"):
                # Reset conversation to initial greeting
                st.session_state.messages = [
                    {"role": "assistant", "content": "Bonjour! Comment puis-je vous aider aujourd'hui?"}
                ]
                # Reload the page to reset the UI
                st.rerun()
    
    # Additional information and features in the sidebar
    with st.sidebar:
        with st.container(border = True):

            # About section with information about the agent
            st.title("❓ À propos")
            st.markdown("""
            
            Cet agent utilise la librairie SmoLAgents pour vous aider à générer l'application streamlit de vos rêves ✨.          
            
            Essayer par vous même ! Vous pouvez demander des visualisations en utilisant des phrases comme:
            - "Montre-moi un graphique en ligne des températures"
            - "Crée un diagramme à barres des ventes par région"
            - "Affiche un nuage de points de l'âge vs revenu"
            
            L'agent détectera automatiquement votre demande et générera une visualisation appropriée.
            """)
        with st.container(border = True):
            st.title("🚧 Aide 🚧")
            st.markdown("""
                - Si l'agent ne répond pas, vérifiez que l'agent est bien connecté.
                - Assurez-vous qu'il vous reste suffisamment de crédit si vous utilisez un agent personnalisé !
                - Essayer de générer une application moins complexe ou d'améliorer votre prompt.""")
            

if __name__ == "__main__":
    main() 