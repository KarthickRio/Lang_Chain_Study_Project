# OpenAI Agents SDK Tracing with LangDB

This example demonstrates how to trace an application built with the OpenAI Agents SDK using LangDB. Full observability is achieved by initializing LangDB and configuring the default OpenAI client to route all model calls through the LangDB proxy.

This sample is based on the [LangDB documentation for the OpenAI Agents SDK](https://docs.langdb.ai/getting-started/working-with-agent-frameworks/working-with-openai-agents-sdk).

## Getting Started

First, clone the repository and navigate to this example:

```bash
git clone https://github.com/langdb/langdb-samples.git
cd langdb-samples/examples/openai/openai-agents-tracing
```

## Prerequisites

1. **LangDB Account**: Ensure you have a LangDB account and have created a project to get your `LANGDB_API_KEY` and `LANGDB_PROJECT_ID`.
2. **Python Environment**: It is recommended to use a virtual environment.

## 1. Setup

Create a `.env` file in this directory (`examples/openai/openai-agents-tracing`) and add your credentials:

```sh
# examples/openai/openai-agents-tracing/.env

LANGDB_API_KEY="<your_langdb_api_key>"
LANGDB_PROJECT_ID="<your_langdb_project_id>"
LANGDB_API_BASE_URL="https://api.us-east-1.langdb.ai" # Or your self-hosted endpoint
```

## 2. Installation

Install the required Python packages:

```sh
pip install -r requirements.txt
```

## 3. How It Works

LangDB provides end-to-end tracing for the OpenAI Agents SDK with minimal configuration:

1. **Initialize Tracing**: The `main.py` script calls `init()` from `pylangdb.openai` at the very beginning. This patches the necessary libraries to capture agent and tool execution data.

    ```python
    from pylangdb.openai import init
    init()
    ```

2. **Configure Default Client for Tracing**: An `AsyncOpenAI` client is configured with your LangDB credentials. This client is then passed to `set_default_openai_client` with `use_for_tracing=True`. This ensures that any model call made using the default client is automatically traced by LangDB.

    ```python
    from agents import set_default_openai_client, Model, OpenAIChatCompletionsModel
    from openai import AsyncOpenAI
    import os

    client = AsyncOpenAI(
        api_key=os.environ["LANGDB_API_KEY"],
        base_url=os.environ["LANGDB_API_BASE_URL"],
        default_headers={"x-project-id": os.environ["LANGDB_PROJECT_ID"]}
    )

    set_default_openai_client(client, use_for_tracing=True)
    ```

3. **Instantiate Models**: Each agent is initialized with a concrete `Model` instance instead of just a model name string. A helper function `get_model` creates instances of `OpenAIChatCompletionsModel`, passing the configured client to them.

    ```python
    def get_model(model_name) -> Model:
        return OpenAIChatCompletionsModel(model=model_name, openai_client=client)

    triage_agent = Agent(
        name="Triage Agent",
        # ...
        model=get_model("openai/gpt-4o-mini")
    )
    ```

This setup automatically links all steps—including model calls, tool usage, and agent handoffs—into a single, comprehensive session trace, which is identified by the `group_id` passed to the `Runner`.

## 4. Usage

Run the example script:

```sh
python main.py
```

The script will execute a multi-agent workflow where a triage agent routes a question to the appropriate specialist agent. After the script completes, you can view the full trace in your LangDB project dashboard.
