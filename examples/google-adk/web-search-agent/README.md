# Google ADK Web Search Agent

A comprehensive 2-step web search agent system built with Google ADK that provides thorough research and synthesized answers to user queries through sequential agent processing.

## Overview

This application uses a two-step sequential agent approach:
- **Critic Agent**: Conducts comprehensive web searches and analyzes information
- **Reviser Agent**: Synthesizes research findings into well-structured, comprehensive answers

The agents work in sequence with a shared thread ID to maintain context throughout the research and synthesis process.

## Environment Setup

Create a `.env` file with your LangDB credentials:

```bash
# LangDB API Configuration
LANGDB_API_KEY="your_langdb_api_key"
LANGDB_PROJECT_ID="your_langdb_project_id"
LANGDB_API_BASE_URL="https://api.us-east-1.langdb.ai"
```

## Initialize LangDB Tracing

To enable automatic tracing of your Google ADK agents, you must initialize `pylangdb` **before** any `google.adk` modules are imported. This is the most critical step for integration.

In `web-search/agent.py`, the `init()` function is called at the top of the file:

```python
# web-search/agent.py
from pylangdb.adk import init

# Initialize LangDB tracing before importing any ADK modules
init()

from google.adk.agents import SequentialAgent
from .sub_agents.critic import critic_agent
# ... rest of the agent setup
```

## Model and Tool Configuration

This agent's search capabilities are enabled through a **LangDB Virtual Model**. Instead of hard-coding tools into the agent, we attach a search tool (Tavily Search) to a model in the LangDB UI. The agent code then only needs to reference the virtual model's name.

This approach decouples the agent's logic from its tools, allowing you to modify tools and models in the LangDB dashboard without changing any code.

### Step 1: Create a Virtual MCP Server

1.  In the LangDB dashboard, go to **MCP Servers → Virtual MCP Servers**.
2.  Click **+ New Virtual MCP Server** and configure it:
    -   **Name**: `web-search-mcp`
    -   **Underlying MCP**: Select the **Tavily Search MCP**.

### Step 2: Create a Virtual Model

1.  Navigate to your project's **Models** page.
2.  Click **+ New Virtual Model** and configure it:
    -   **Name**: `critic-agent-model`
    -   **Base Model**: Select a base model (e.g., `openai/gpt-4o`).
    -   **MCP Server**: Attach the `web-search-mcp` you created.
3.  Save the model. LangDB will generate a unique name for it, such as `langdb/critic-agent-model_xxxxxx`.

### Step 3: Use the Virtual Model in the Agent

In `web-search/sub_agents/critic/agent.py`, update the `Agent` to use the new virtual model name. 

```python
# web-search/sub_agents/critic/agent.py
from google.adk.agents import Agent

critic_agent = Agent(
    # Use the virtual model name from the LangDB UI
    model="langdb/critic-agent-model_xxxxxx", 
    name="critic_agent",
    instruction=prompt.CRITIC_PROMPT,
    after_model_callback=_render_reference # Optional: for formatting results
)
```

With this setup, all calls made by the `critic_agent` are automatically routed through LangDB, enabling full traceability of model inputs, outputs, and tool calls.

## Architecture

The web search agent is built on **Google ADK's SequentialAgent** architecture with two specialized sub-agents, automatically traced by LangDB:

1. **Critic Agent**
   - Analyzes user queries to understand information needs
   - Conducts comprehensive web searches using the attached Tavily Search tool via its Virtual Model.
   - Evaluates source reliability and information quality
   - Organizes findings and identifies information gaps

2. **Reviser Agent**
   - Receives research findings from the Critic Agent
   - Synthesizes information into coherent, comprehensive answers
   - Structures responses with appropriate formatting and organization
   - Ensures final answers directly address user queries

Both agents use Google ADK's threading system with a shared thread ID to maintain context throughout the process.

## Key Features

### 1. Sequential Processing
- Two-step approach ensures thorough research followed by quality synthesis
- Shared thread ID maintains context between agents

### 2. Comprehensive Web Search
- Multiple search strategies and varied search terms
- Focus on authoritative and recent sources
- Analysis of conflicting information and source reliability

### 3. Direct MCP Server Integration
- Web search capabilities through Tavily Search MCP
- Direct MCP server configuration in LangDBLlm
- Simplified setup without additional toolsets

### 4. Professional Synthesis
- Well-structured, comprehensive final answers
- Acknowledgment of limitations and conflicting information
- Clear, accessible writing style with proper organization

### 5. Model Flexibility
- Google ADK foundation with LangDB ADK extension supports multiple model providers
- Configurable model selection per agent via LangDB integration
- Direct MCP server integration through LangDB for enhanced capabilities

## Usage

### Example Usage

```bash
adk run web-search
```
Or 
```bash
adk web
```

## Project Structure

```
web-search-agent/
├── web-search/
│   ├── __init__.py
│   ├── agent.py              # Main SequentialAgent configuration
│   └── sub_agents/
│       ├── __init__.py
│       ├── critic/           # Web search and analysis agent
│       │   ├── __init__.py
│       │   ├── agent.py      # Critic agent with MCP tools
│       │   └── prompt.py     # Research and analysis prompts
│       └── reviser/          # Synthesis and formatting agent
│           ├── __init__.py
│           ├── agent.py      # Reviser agent configuration
│           └── prompt.py     # Synthesis and writing prompts
├── pyproject.toml            # Project dependencies
└── README.md           # This file
```

## Dependencies

### Prerequisites

- Python 3.11+
- [UV](https://github.com/astral-sh/uv) - A fast Python package installer.

### Installation

1.  Install project dependencies using UV:
    ```bash
    uv pip install -e .
    ```

### Main Dependencies

- `google-adk`: Google's Agents Development Kit for building multi-agent systems.
- `pylangdb`: Provides automatic tracing and integration for Google ADK.

## References

* [Building Web Search Agent with Google-ADK](https://docs.langdb.ai/guides/building-agents/building-web-search-agent-with-google-adk)
* [Google ADK Documentation](https://google.github.io/adk-docs/)
* [LangDB Virtual Models](https://docs.langdb.ai/concepts/virtual-models)