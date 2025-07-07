# Agno Basic Example with LangDB Tracing

This example demonstrates how to trace a basic Agno agent using LangDB. Tracing is enabled through a combination of LangDB's `init()` function and the `agno.models.langdb.LangDB` model class, which routes all model interactions through the LangDB proxy.

This sample is based on the [LangDB documentation for Agno](https://docs.langdb.ai/getting-started/working-with-agent-frameworks/working-with-agno).

## Getting Started

First, clone the repository and navigate to this example:

```bash
git clone https://github.com/langdb/langdb-samples.git
cd langdb-samples/examples/agno/agno-basic
```

## Prerequisites

1. **LangDB Account**: Ensure you have a LangDB account and have created a project to get your `LANGDB_API_KEY` and `LANGDB_PROJECT_ID`.
2. **Python Environment**: It is recommended to use a virtual environment.

## 1. Setup

Create a `.env` file in this directory (`examples/agno/agno-basic`) and add your credentials:

```sh
# examples/agno/agno-basic/.env

LANGDB_API_KEY="<your_langdb_api_key>"
LANGDB_PROJECT_ID="<your_langdb_project_id>"
```

## 2. Installation

Install the required Python packages:

```sh
pip install -r requirements.txt
```

## 3. How It Works

LangDB uses a two-part approach for comprehensive tracing with Agno:

### a) Structural Tracing with `init()`

The `main.py` script calls `init()` from `pylangdb.agno` at the very beginning. This instruments the Agno library to capture the high-level structure of your agent's execution.

```python
from pylangdb.agno import init
init()
```

### b) Model Call Tracing with `LangDB` Model

To trace the LLM calls, the agent is configured to use the `agno.models.langdb.LangDB` model. This class directs all model requests through the LangDB proxy, capturing the prompts, responses, token usage, and other metrics.

```python
import os
from agno.agent import Agent
from agno.models.langdb import LangDB

# Configure LangDB-backed model
langdb_model = LangDB(
    id="openai/gpt-4",
    api_key=os.getenv("LANGDB_API_KEY"),
    project_id=os.getenv("LANGDB_PROJECT_ID"),
)

# Create and run your agent
agent = Agent(
    model=langdb_model,
    # ...
)
```

This combination ensures that both the agent's structure and its interactions with the LLM are fully traced.

## 4. Usage

Run the example script:

```sh
python main.py
```

After the script completes, you will see the full trace of the Agno agent's execution in your LangDB project dashboard.
