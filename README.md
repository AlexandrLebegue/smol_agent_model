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

# SmoLAgents Conversational Agent

A powerful conversational agent built with SmoLAgents that can connect to various language models, perform web searches, create visualizations, execute code, and much more.

## 📋 Overview

This project provides a flexible and powerful conversational agent that can:

- Connect to different types of language models (local or cloud-based)
- Perform web searches to retrieve up-to-date information
- Visit and extract content from webpages
- Execute shell commands with appropriate security measures
- Create and modify files
- Generate data visualizations based on natural language requests
- Execute Python code within the chat interface

The agent is available through two interfaces:
- A Gradio interface (original)
- A Streamlit interface (new) with enhanced features and configuration options

## 🛠️ Prerequisites

- Python 3.8+
- A language model, which can be one of:
  - A local model running through an OpenAI-compatible API server (like [LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.ai/), etc.)
  - A Hugging Face model accessible via API
  - A cloud-based model with API access

## 🚀 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/smolagents-conversational-agent.git
   cd smolagents-conversational-agent
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🔧 Setup

### Setting Up a Language Model

You have several options for the language model:

#### Option 1: Local Model with LM Studio (Recommended for beginners)

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Launch LM Studio and download a model (e.g., Mistral 7B, Llama 2, etc.)
3. Start the local server by clicking "Start Server"
4. Note the server URL (typically http://localhost:1234/v1)

#### Option 2: Using OpenRouter

1. Create an account on [OpenRouter](https://openrouter.ai/)
2. Get your API key from the dashboard
3. Use the OpenRouter URL and your API key in the agent configuration

#### Option 3: Hugging Face API ( no more tested be careful )

1. If you have access to Hugging Face API endpoints, you can use them directly
2. Configure the URL and parameters in the agent interface

## 💻 Usage

### Streamlit Interface (Recommended)

The Streamlit interface offers a more user-friendly experience with additional features:

1. Launch the Streamlit application:
   ```bash
   streamlit run streamlit_app.py
   ```

2. Access the interface in your web browser at http://localhost:8501

3. Configure your model in the sidebar:
   - Select the model type (OpenAI Server, Hugging Face API, or Hugging Face Cloud)
   - Enter the required configuration parameters
   - Click "Apply Configuration"

4. Start chatting with the agent in the main interface

### Gradio Interface

The original Gradio interface is still available:

1. Launch the Gradio application:
   ```bash
   python app.py
   ```

2. Access the interface in your web browser at the URL displayed in the terminal (typically http://localhost:7860)

## 🌟 Features

### Streamlit Interface Features

- **Interactive Chat Interface**: Engage in natural conversations with the agent
- **Multiple Model Support**:
  - OpenAI Server (LM Studio or other OpenAI-compatible servers)
  - Hugging Face API
  - Hugging Face Cloud
- **Real-time Agent Reasoning**: See the agent's thought process as it works on your request
- **Customizable Configuration**: Adjust model parameters without modifying code
- **Data Visualization**: Request and generate charts directly in the chat
- **Code Execution**: Run Python code generated by the agent within the chat interface
- **Timezone Display**: Check current time in different time zones

### Agent Tools

The agent comes equipped with several powerful tools:

- **Web Search**: Search the web via DuckDuckGo to get up-to-date information
- **Webpage Visiting**: Visit and extract content from specific webpages
- **Shell Command Execution**: Run commands on your system (with appropriate security)
- **File Operations**: Create and modify files on your system
- **Data Visualization**: Generate charts and graphs based on your requests
- **Code Execution**: Run Python code within the chat interface

## 🧩 Extending the Agent

### Adding Custom Tools

You can extend the agent with your own custom tools by modifying the `app.py` file:

```python
@tool
def my_custom_tool(arg1: str, arg2: int) -> str:
    """Description of what the tool does
    Args:
        arg1: description of the first argument
        arg2: description of the second argument
    """
    # Your tool implementation
    return "Tool result"
```

### Customizing Prompts

The agent's behavior can be customized by modifying the prompt templates in the `prompts.yaml` file.

## 📊 Visualization Examples

The agent can generate visualizations based on natural language requests. Try asking:

- "Show me a line chart of temperature trends over the past year"
- "Create a bar chart of sales by region"
- "Display a scatter plot of age vs. income"

## 🔍 Troubleshooting

- **Agent not responding**: Verify that your LLM server is running and accessible
- **Connection errors**: Check the URL and API key in your configuration
- **Slow responses**: Consider using a smaller or more efficient model
- **Missing dependencies**: Ensure all requirements are installed via `pip install -r requirements.txt`

## 📚 Examples

Here are some example queries you can try with the agent:

- "What's the current time in Tokyo?"
- "Can you summarize the latest news about AI?"
- "Create a Python function to sort a list of dictionaries by a specific key"
- "Explain how transformer models work in AI"
- "Show me a bar chart of population by continent"
- "Write a simple web scraper to extract headlines from a news website"

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

*For more information on Hugging Face Spaces configuration, visit https://huggingface.co/docs/hub/spaces-config-reference*
