# Google ADK Travel Concierge Agent

A sophisticated, multi-agent system built with Google ADK that provides end-to-end travel planning, from inspiration to post-trip feedback, all with full observability through LangDB.

## Getting Started

First, clone the repository and navigate to this example:

```bash
git clone https://github.com/langdb/langdb-samples.git
cd langdb-samples/examples/google-adk/travel-concierge
```

## Overview

This application demonstrates a hierarchical agent system where a main `root_agent` orchestrates a team of specialized sub-agents to handle a complete travel journey:

-   **Inspiration Agent**: Helps users discover travel destinations.
-   **Planning Agent**: Creates detailed itineraries.
-   **Booking Agent**: Assists with booking flights and hotels.
-   **Pre-Trip Agent**: Provides preparation guidance.
-   **In-Trip Agent**: Offers real-time assistance during travel.
-   **Post-Trip Agent**: Gathers feedback and helps with post-travel tasks.

This modular design allows each agent to focus on a specific part of the travel process, with all interactions traced and managed through the LangDB AI Gateway.

## Environment Setup

Create a `.env` file in the project's root directory with your LangDB credentials:

```bash
# LangDB API Configuration
LANGDB_API_KEY="your_langdb_api_key"
LANGDB_PROJECT_ID="your_langdb_project_id"
```

## Initialize LangDB Tracing

To enable automatic, end-to-end tracing of your Google ADK agents, you must initialize `pylangdb` **before** any `google.adk` modules are imported. This is the most critical step for the integration.

In `travel_concierge/agent.py`, the `init()` function is called at the top of the file:

```python
# travel_concierge/agent.py
from pylangdb.adk import init

# Initialize LangDB tracing before importing any ADK modules
init()

from google.adk.agents import Agent
from .sub_agents.inspiration import inspiration_agent
# ... rest of the agent setup
```

This single line of code enables LangDB to automatically discover all agents, patch the ADK runtime, and link all interactions into a single, cohesive trace.

## Model and Tool Configuration

This agent's powerful capabilities (search, booking, etc.) are enabled through **LangDB Virtual Models** and **Virtual MCP Servers**. Instead of hard-coding tools into the agent, we attach them to models in the LangDB UI. The agent code then only needs to reference the virtual model's name.

This decouples the agent's logic from its tools, allowing you to add, remove, or swap tools in the LangDB dashboard without changing any code.

### Step 1: Create Virtual MCP Servers

In the LangDB dashboard, create Virtual MCP Servers for the external services you need. For this agent, you might create:

-   A `google-maps-mcp` for the **Inspiration Agent**.
-   An `airbnb-mcp` for the **Planning Agent**.
-   A `tavily-search-mcp` for **Google Search Agent in Tools**.

### Step 2: Create Virtual Models

1.  Navigate to your project's **Models** page in the LangDB UI.
2.  Click **+ New Virtual Model** for each agent that needs tools. For example, for the Inspiration Agent:
    -   **Name**: `travel-inspiration-model`
    -   **Base Model**: Select a powerful base model (e.g., `anthropic/claude-3.5-sonnet`).
    -   **MCP Server**: Attach the `google-maps-mcp` you created.
3.  Save the model. LangDB will generate a unique name for it, such as `langdb/travel-inspiration-model_xxxxxx`.

### Step 3: Use the Virtual Model in the Agent

In the agent's definition file (e.g., `travel_concierge/sub_agents/inspiration/agent.py`), update the `Agent` to use the new virtual model name.

```python
# travel_concierge/sub_agents/inspiration/agent.py
from google.adk.agents import Agent

inspiration_agent = Agent(
    # Use the virtual model name from the LangDB UI
    model="langdb/travel-inspiration-model_xxxxxx",
    name="inspiration_agent",
    instruction=prompt.INSPIRATION_PROMPT,
)
```

With this setup, all calls made by the `inspiration_agent` are automatically routed through LangDB, enabling full traceability and dynamic tool use.

## Architecture

The Travel Concierge is built on **Google ADK's hierarchical agent** architecture, automatically traced by LangDB:

1.  **Root Agent**: The central orchestrator. It receives user queries and delegates them to the appropriate sub-agent based on the user's intent. It does not have tools of its own.
2.  **Sub-Agents**: Each sub-agent is a specialist.
    -   **Inspiration Agent**: Uses tools like Google Maps and Tavily Search to provide travel ideas.
    -   **Planning Agent**: Uses tools like Airbnb or flight search APIs to build itineraries.
    -   ...and so on for booking, pre-trip, in-trip, and post-trip tasks.

This structure creates a clean separation of concerns, making the system scalable and easy to maintain.

## Key Features

### 1. Hierarchical Delegation
- A central root agent routes tasks to specialized sub-agents.
- Maintains context across multiple turns and agents.

### 2. Dynamic Tooling with Virtual MCPs
- Tools (APIs, services) are managed in the LangDB UI, not in the code.
- Add or change agent capabilities without redeploying the application.
- API keys are stored securely in LangDB.

### 3. End-to-End Tracing
- `pylangdb.adk.init()` provides zero-instrumentation tracing.
- Get complete visibility into every agent interaction, model call, and tool execution in the LangDB dashboard.

### 4. Model Flexibility
- Assign different LLMs to different agents based on their tasks.
- Swap models easily through the LangDB UI.

## Usage

### Run the Agent from the Command Line

```bash
adk run travel-concierge
```

### Run the Web Interface

```bash
adk web
```
Navigate to `http://localhost:8000`, select `travel_concierge`, and start planning your trip.

## Project Structure

```
travel-concierge/
├── travel_concierge/
│   ├── __init__.py
│   ├── agent.py              # Main root_agent and sub-agent orchestration
│   ├── prompt.py             # Prompts for the root agent
│   └── sub_agents/
│       ├── __init__.py
│       ├── inspiration/      # Travel inspiration agent
│       │   ├── __init__.py
│       │   ├── agent.py
│       │   └── prompt.py
│       ├── planning/         # Itinerary planning agent
│       │   ├── __init__.py
│       │   ├── agent.py
│       │   └── prompt.py
│       └── ...               # Other agents (booking, pre-trip, etc.)
├── pyproject.toml            # Project dependencies
└── README.md                 # This file
```

## Installation

### Prerequisites

- Python 3.11+
- [UV](https://github.com/astral-sh/uv) - A fast Python package installer (recommended)

### Option 1: Quick Install

Install the required packages directly using pip:

```bash
pip install google-adk "pylangdb[adk]" python-dotenv
```

### Option 2: Development Install

If you've cloned the repository and want to install it in development mode:

```bash
# Using pip
pip install -e .

# Or using UV (recommended for better dependency resolution)
uv pip install -e .
```

### Main Dependencies

- `google-adk`: Google's Agents Development Kit for building multi-agent systems
- `pylangdb[adk]`: Provides automatic tracing and integration for Google ADK
- `python-dotenv`: For loading environment variables from .env files

## References

* [Building Travel Concierge with Google-ADK](https://docs.langdb.ai/guides/building-agents/building-travel-concierge-with-google-adk)
* [Discover End-to-End Tracing on Google ADK with LangDB](https://blog.langdb.ai/discover-end-to-end-tracing-on-google-adk-with-langdb)
* [Google ADK Documentation](https://google.github.io/adk-docs/)
* [LangDB Virtual Models](https://docs.langdb.ai/concepts/virtual-models)