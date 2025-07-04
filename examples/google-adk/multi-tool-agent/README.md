# Google ADK Multi-Tool Agent with LangDB Tracing

This example demonstrates how to trace a Google Agent Development Kit (ADK) agent that uses multiple tools. LangDB's `init()` function automatically instruments the ADK to capture and send traces to your LangDB project.

This sample is based on the official [Google ADK Quickstart](https://google.github.io/adk-docs/get-started/quickstart/) and the [LangDB documentation for Google ADK](https://docs.langdb.ai/getting-started/working-with-agent-frameworks/working-with-google-adk).

## Prerequisites

1. **LangDB Account**: Ensure you have a LangDB account and have created a project to get your `LANGDB_API_KEY` and `LANGDB_PROJECT_ID`.
2. **Python Environment**: It is recommended to use a virtual environment.

## 1. Setup

Create a `.env` file in this directory (`examples/google-adk/multi-tool-agent`) and add your credentials:

```sh
# examples/google-adk/multi-tool-agent/.env

LANGDB_API_KEY="YOUR_LANGDB_API_KEY"
LANGDB_PROJECT_ID="YOUR_LANGDB_PROJECT_ID"
```

## 2. Installation

Install the required Python packages:

```sh
pip install -r requirements.txt
```

This will install `pylangdb` with the `adk` extra, which includes the Google ADK, `litellm`, and other dependencies.

## 3. How It Works

The `multi_tool_agent/agent.py` script initializes LangDB tracing *before* importing any ADK modules:

```python
# First initialize LangDB before defining any agents
from pylangdb.adk import init
init()

# All other imports, including google.adk, come after init()
import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
# ...
```

This `init()` call enables automatic tracing for all subsequent ADK operations. LangDB automatically discovers all agents and sub-agents, wraps their key methods at runtime, and links sessions for full end-to-end tracing.

## 4. Usage

Navigate to the parent directory of the agent (`examples/google-adk/multi-tool-agent`) and start the ADK web interface:

```sh
adk web
```

Open the URL provided (usually `http://localhost:8000`) in your browser and select `multi_tool_agent` from the dropdown menu.

Once your agent is running, try these example queries to test its functionality:

* `What's the weather in New York?`
* `What is the time in New York?`

These queries will trigger the agent to use the defined tools. After each interaction, you will see the full trace in your LangDB project dashboard.
