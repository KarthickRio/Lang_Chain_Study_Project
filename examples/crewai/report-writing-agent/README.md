# CrewAI Report Writing Agent

A multi-agent report generation system built with CrewAI that creates comprehensive research reports on any given topic.

## Overview

This application uses three specialized AI agents working in sequence:
- **Researcher**: Gathers comprehensive information on the topic
- **Analyst**: Analyzes and synthesizes the research findings
- **Report Writer**: Creates a professional Markdown report

## Environment Setup

### Required Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# LangDB API Configuration
LANGDB_API_KEY=your_api_key_here
LANGDB_API_BASE_URL=https://api.us-east-1.langdb.ai
LANGDB_PROJECT_ID=your_project_id_here
```

### Environment Variable Details

- `LANGDB_API_KEY`: Your LangDB API key (required)
- `LANGDB_API_BASE_URL`: The base URL for the LangDB API (e.g., `https://api.us-east-1.langdb.ai`)
- `LANGDB_PROJECT_ID`: Your LangDB project identifier

### Initialize LangDB Tracing

Before any CrewAI code runs, initialize LangDB tracing with `pylangdb`:

```python
from pylangdb.crewai import init
from dotenv import load_dotenv

load_dotenv()  # loads the .env variables set above
init()         # start tracing
```

## Model Configuration

This project uses a flexible model configuration system, allowing you to assign different models to each agent. Models are configured in `main.py` using the `create_llm()` helper function.

### How Models are Configured

In `main.py`, each agent's `llm` is instantiated by calling `create_llm()` with a model name string.

```python
# Example from main.py
llm = create_llm("openai/gpt-4o", "analysis")
```

To change the model for any agent, simply modify the model string in the corresponding agent method:

- **Researcher Agent**: Edit the `create_llm()` call inside the `researcher` method.
- **Analyst Agent**: Edit the `create_llm()` call inside the `analyst` method.
- **Report Writer Agent**: Edit the `create_llm()` call inside the `report_writer` method.

### Supported Model Names

LangDB supports a wide range of models from different providers using the LiteLLM format: `provider/model_name`.

Here are some examples of valid model names:
- `openai/gpt-4o`
- `openai/gpt-3.5-turbo`
- `anthropic/claude-3.5-sonnet-20240620`
- `google/gemini-1.5-pro-latest`

### Using Virtual Models for Tools (Researcher Agent)

The **Researcher Agent** needs to use the Tavily Search tool to gather real-time information. To enable this, we need to configure a **Virtual Model** in LangDB that has the search tool attached.

This involves two steps:
1.  Creating a Virtual MCP Server for the search tool.
2.  Creating a Virtual Model that uses this MCP Server.

#### 1. Virtual MCP Server Setup

1.  Log in to your LangDB account and navigate to **MCP Servers → Virtual MCP Servers**.
2.  Click **+ New Virtual MCP Server** and configure it:
    -   **Name**: Give it a descriptive name (e.g., `web-search-mcp`).
    -   **Underlying MCP**: Choose the **Tavily Search MCP** from the list.
3.  Save the new server.

#### 2. Virtual Model Setup

1.  Navigate to your project's **Models** page.
2.  Click **+ New Virtual Model**.
3.  Configure the model:
    -   **Name**: e.g., `report-researcher`.
    -   **Base Model**: Choose a powerful model for the agent (e.g., `openai/gpt-4o`).
    -   **MCP Server**: Select the `web-search-mcp` you created in the previous step.
4.  After creating the model, LangDB will generate a unique model name, like `openai/langdb/report-researcher@v1`.
5.  Copy this name and use it in `main.py` for the `researcher` agent:

```python
# In main.py, inside the ReportGenerationCrew class
@agent
def researcher(self) -> Agent:
    return Agent(
        config=self.agents_config['researcher'],
        verbose=True,
        # Use the virtual model name from LangDB
        llm=create_llm("openai/langdb/report-researcher@v1", "research")
    )
```

This setup ensures that when the `researcher` agent runs, it has access to the Tavily search tool, and all its search activities are traced in LangDB. 


## Usage

### Command Line

```bash
# Run with topic as argument
python main.py "Your research topic here"

# Run interactively (will prompt for topic)
python main.py
```

### Example

```bash
python main.py "The Impact of Artificial Intelligence on Social Media Marketing: A 2024 Report"
```

## Output

The application generates:
- Console output showing the progress of each agent
- A `report.md` file containing the final report
- Preview of the report in the terminal

## Project Structure

```
main.py                 # Main application entry point
configs/
   agents.yaml        # Agent configurations (roles, goals, backstories)
   tasks.yaml         # Task configurations and workflows
pyproject.toml         # Project dependencies and metadata
README.md              # This file
```

## Agent Configuration

Agents are configured in `configs/agents.yaml`:
- **researcher**: Domain expert focused on gathering current information
- **analyst**: Data specialist for synthesizing insights
- **report_writer**: Technical writer for creating structured reports

## Task Configuration

Tasks are defined in `configs/tasks.yaml`:
- **research_task**: Comprehensive topic research
- **analysis_task**: Data analysis and insight extraction
- **report_writing_task**: Professional report creation

## Dependencies

### Prerequisites

- Python 3.8+
- [UV](https://github.com/astral-sh/uv) - Fast Python package installer and resolver

### Installation

1. First, install UV (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Follow the on-screen instructions to add UV to your PATH.

2. Install project dependencies using UV:
   ```bash
   uv pip install -e .
   ```
   
   This will install all the required packages listed in `pyproject.toml`.

### Main Dependencies

- `crewai>=0.108.0`: Multi-agent orchestration framework
- `crewai-tools>=0.0.1`: Additional tools for CrewAI
- `python-dotenv>=1.0.1`: Environment variable management
- `pylangdb>=0.2.0`: LangDB tracing integration for CrewAI


## References

* [LangDB Virtual MCP Servers](https://docs.langdb.ai/concepts/virtual-mcp-servers)
* [LangDB Virtual MCP](https://docs.langdb.ai/concepts/virtual-mcp-servers)
* [CrewAI Documentation](https://docs.crewai.com/)