# LangGraph Tracing with LangDB

This example demonstrates how to trace a LangGraph agent using LangDB. Full observability is achieved by combining LangDB's `init()` function for structural tracing with the configuration of the `ChatOpenAI` client to route model calls through the LangDB proxy.

This sample is based on the [LangDB documentation for LangGraph](https://docs.langdb.ai/getting-started/working-with-agent-frameworks/working-with-langgraph).

## Prerequisites

1. **LangDB Account**: Ensure you have a LangDB account and have created a project to get your `LANGDB_API_KEY` and `LANGDB_PROJECT_ID`.
2. **Python Environment**: It is recommended to use a virtual environment.

## 1. Setup

Create a `.env` file in this directory (`examples/langchain/langgraph-tracing`) and add your credentials:

```sh
# examples/langchain/langgraph-tracing/.env

LANGDB_API_KEY="<your_langdb_api_key>"
LANGDB_PROJECT_ID="<your_langdb_project_id>"
LANGDB_API_BASE_URL='https://api.us-east-1.langdb.ai'
```

## 2. Installation

Install the required Python packages:

```sh
pip install -r requirements.txt
```

## 3. How It Works

LangDB uses a two-part approach to provide comprehensive tracing for LangGraph:

### Structural Tracing with `init()`

The `main.py` script calls `init()` from `pylangdb.langchain` at the very beginning. This instruments the LangChain and LangGraph libraries to capture the high-level structure of your graph, including nodes, edges, and tool calls.

```python
from pylangdb.langchain import init
init()
```

### Model Call Tracing with `ChatOpenAI` Configuration

To trace the LLM interactions, the `ChatOpenAI` client is configured to route requests through the LangDB proxy. This is done by setting the `openai_api_base`, `openai_api_key`, and `default_headers` parameters.

The `main.py` example uses a helper function for this:

```python
import os
from langchain_openai import ChatOpenAI

def create_model():
    api_base = os.getenv("LANGDB_API_BASE_URL") # Defaults to LangDB proxy
    api_key = os.getenv("LANGDB_API_KEY")
    project_id = os.getenv("LANGDB_PROJECT_ID")
    default_headers = {
        "x-project-id": project_id,
    }
    llm = ChatOpenAI(
        model_name='openai/gpt-4o',
        openai_api_base=api_base,
        openai_api_key=api_key,
        default_headers=default_headers
    )
    return llm.bind_tools([get_weather_forecast])
```

This combination ensures that both the graph's structure and its interactions with the LLM are fully traced.

## Usage

Run the example script:

```sh
python main.py
```

After the script completes, you will see the full, detailed trace of the LangGraph execution in your LangDB project dashboard.
