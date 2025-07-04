# CrewAI Tracing with LangDB

This example demonstrates how to trace CrewAI agent and task executions using LangDB. This is achieved through a combination of LangDB's `init()` function for structural tracing and configuring the CrewAI `LLM` to use the LangDB proxy for model call tracing.

This sample is based on the [LangDB documentation for CrewAI](https://docs.langdb.ai/getting-started/working-with-agent-frameworks/working-with-crewai).

## Prerequisites

1. **LangDB Account**: Ensure you have a LangDB account and have created a project to get your `LANGDB_API_KEY` and `LANGDB_PROJECT_ID`.
2. **Python Environment**: It is recommended to use a virtual environment.

## 1. Setup

Create a `.env` file in this directory (`examples/crewai/crewai-tracing`) and add your credentials:

```sh
# examples/crewai/crewai-tracing/.env

LANGDB_API_KEY="<your_langdb_api_key>"
LANGDB_PROJECT_ID="<your_langdb_project_id>"

# Optional: If you are self-hosting or using a different region
# LANGDB_API_BASE_URL="<your_langdb_api_base_url>"
```

You will also need to set `SERPER_API_KEY` in the `.env` file to use the `SerperDevTool` included in the example.

## 2. Installation

Install the required Python packages:

```sh
pip install -r requirements.txt
```

This will install `pylangdb` with the `crewai` extra, which includes `crewai`, `crewai-tools`, and other dependencies.

## 3. How It Works

LangDB uses a two-part approach to provide comprehensive tracing for CrewAI:

### a) Structural Tracing with `init()`

The `main.py` script initializes LangDB tracing *before* importing any CrewAI modules. This call instruments CrewAI to capture the high-level structure of your agents, tasks, and crew.

```python
from pylangdb.crewai import init
init()

# All other imports, including crewai, come after init()
from crewai import Agent, Task, Crew, LLM
# ...
```

### b) Model Call Tracing with `LLM` Configuration

To trace the actual Large Language Model (LLM) calls, you must configure the CrewAI `LLM` object to route requests through the LangDB proxy. This is done by setting the `api_key`, `base_url`, and `extra_headers`.

The `main.py` example uses a helper function for this:

```python
import os
from crewai import LLM

def create_llm(model):
    return LLM(
        model=model,
        api_key=os.environ.get("LANGDB_API_KEY"),
        base_url=os.environ.get("LANGDB_API_BASE_URL"), # Defaults to LangDB proxy
        extra_headers={"x-project-id": os.environ.get("LANGDB_PROJECT_ID")}
    )

# Example usage when creating an agent
researcher = Agent(
    role="Research Specialist",
    goal="Research topics thoroughly",
    llm=create_llm("openai/gpt-4o"),
    # ...
)
```

By combining these two steps, LangDB provides a full end-to-end trace of your CrewAI workflow.

## 4. Usage

Run the example script with a topic:

```sh
python main.py "Artificial Intelligence in Healthcare"
```

After the script completes, you will see the full trace of the CrewAI execution, including all agent, task, and tool interactions, in your LangDB project dashboard.
