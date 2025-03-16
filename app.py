from smolagents import CodeAgent,DuckDuckGoSearchTool, HfApiModel,load_tool,tool
import datetime
import requests
import pytz
import yaml
import os
import sys 
import subprocess  # Ajout de l'import manquant pour ShellCommandTool
import io
import json
from huggingface_hub import HfApi
from tools.final_answer import FinalAnswerTool
from tools.visit_webpage import VisitWebpageTool
from tools.web_search import DuckDuckGoSearchTool
from Gradio_UI import GradioUI
from smolagents.models import OpenAIServerModel
from tools.create_file_tool import CreateFileTool
from tools.modify_file_tool import ModifyFileTool

# Below is an example of a tool that does nothing. Amaze us with your creativity !
@tool
def my_custom_tool(arg1:str, arg2:int)-> str: #it's import to specify the return type
    #Keep this format for the description / args / args description but feel free to modify the tool
    """A tool that does nothing yet 
    Args:
        arg1: the first argument
        arg2: the second argument
    """
    return "What magic will you build ?"

# Below is an example of a tool that does nothing. Amaze us with your creativity !
@tool
def get_current_realtime()-> str: #it's import to specify the return type
    #Keep this format for the description / args / args description but feel free to modify the tool
    """A tool that get the current realtime
    """
    return datetime.datetime.now()
@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' è

# model = HfApiModel(
#     model_id="http://192.168.1.141:1234/v1",
#     max_new_tokens=2096,
#     temperature=0.5
# )
# Configuration du modèle pour se connecter au LLM hébergé localement via LMStudio
model = OpenAIServerModel(
    api_base ="http://192.168.1.141:1234/v1",
    model_id="Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",  # Nom arbitraire pour le modèle local
    api_key="sk-dummy-key"  # Clé factice pour LMStudio
    # max_tokens=2096,

)

# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
# Tentative de correction pour ShellCommandTool
try:
    from tools.shell_tool import ShellCommandTool
    shell_tool = ShellCommandTool()
except Exception as e:
    print(f"Erreur lors du chargement de ShellCommandTool: {e}")
    # Créer une version simplifiée de l'outil si nécessaire
    shell_tool = None

agent = CodeAgent(
    model=model,
    tools=[final_answer, DuckDuckGoSearchTool(), VisitWebpageTool(), CreateFileTool(), ModifyFileTool()],
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)

# Ajouter ShellCommandTool conditionnellement
if shell_tool is not None:
    agent.tools['shell_command'] = shell_tool

# Sauvegarder manuellement sans utiliser to_dict() pour éviter les erreurs de validation
agent_data = {
    "name": agent.name,
    "description": agent.description,
    "model": agent.model.to_dict() if hasattr(agent.model, "to_dict") else str(agent.model),
    "tools": [tool.__class__.__name__ for tool in agent.tools.values()],
    "max_steps": agent.max_steps,
    "grammar": agent.grammar,
    "planning_interval": agent.planning_interval,
}

# # Sauvegarder l'agent au format JSON personnalisé
# with open("agent.json", "w", encoding="utf-8") as f:
#     json.dump(agent_data, f, ensure_ascii=False, indent=2)

# # La méthode push_to_hub pose problème avec les emojis, utiliser plutôt le script push_to_hf.py
# print("Agent sauvegardé dans agent.json. Utilisez push_to_hf.py pour le pousser sur Hugging Face.")

# Utiliser l'API Hugging Face directement avec encodage UTF-8
# try:
#     api = HfApi()
#     api.upload_file(
#         path_or_fileobj="agent.json",
#         path_in_repo="agent.json",
#         repo_id="KebabLover/SmolCoderAgent_0_1",
#         repo_type="space",
#         commit_message="Mise à jour de l'agent"
#     )
#     print("Agent poussé avec succès vers Hugging Face!")
# except Exception as e:
#     print(f"Erreur lors du push vers Hugging Face: {e}")

GradioUI(agent).launch()