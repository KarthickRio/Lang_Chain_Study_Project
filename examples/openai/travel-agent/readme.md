# OpenAI Agents SDK Example: Multi-Agent Travel Workflow

This guide illustrates how to build a multi-agent travel query workflow using the OpenAI Agents SDK, augmented by LangDB for advanced tracing, tool integration, and model routing.

We will create a 4-agent pipeline:
1.  **Query Router Agent**: Routes user queries to the appropriate specialist agent.
2.  **Booking Specialist**: Manages booking-related requests.
3.  **Travel Recommendation Specialist**: Provides destination recommendations with web search support.
4.  **Reply Agent**: Formats the final output for the user.

---

## 1. Environment Setup

Create a `.env` file and add your LangDB credentials. These will be used to automatically configure tracing and route model calls through the LangDB gateway.

```bash
# .env
LANGDB_API_KEY="<your_langdb_api_key>"
LANGDB_PROJECT_ID="<your_langdb_project_id>"
LANGDB_API_BASE_URL="https://api.us-east-1.langdb.ai"
```

## 2. Installation

Install the necessary packages. `pylangdb[openai]` provides automatic integration with the OpenAI SDK.

```bash
pip install 'pylangdb[openai]' openai-agents python-dotenv
```

---

## 3. Code Walkthrough

### Initialize LangDB Tracing

In your application entry point (`app.py`), import and call `pylangdb.openai.init()` **before any other imports**. This patches the OpenAI library to ensure all subsequent operations are traced.

```python
# app.py
import os
from dotenv import load_dotenv
from pylangdb.openai import init

# Load environment variables and initialize tracing
load_dotenv()
init()

# ... rest of your application code
```

### Configure the OpenAI Client

Next, create an `AsyncOpenAI` client. `pylangdb` automatically uses the environment variables you set to configure the client for LangDB, so no manual header configuration is needed.

```python
# app.py
from agents import (
    Agent,
    Runner,
    set_default_openai_client,
    RunConfig,
    OpenAIChatCompletionsModel
)
from openai import AsyncOpenAI

# Client is automatically configured by pylangdb.init()
client = AsyncOpenAI()

# Set the configured client as the default for the Agents SDK
set_default_openai_client(client, use_for_tracing=True)

# Helper to create model instances for agents
def get_model(model_name):
    return OpenAIChatCompletionsModel(model=model_name, openai_client=client)
```

### Define the Agents

Now, define your agents. The `model` parameter for each agent should be the name of a **LangDB Virtual Model**. This allows you to attach tools (like web search) and guardrails in the LangDB UI without changing your code.

```python
# app.py

# Define specialist agents
booking_agent = Agent(
    name="Booking Specialist",
    instructions="You are a booking specialist...",
    model=get_model("openai/gpt-4o-mini") # Can be a base model
)

travel_recommendation_agent = Agent(
    name="Travel Recommendation Specialist",
    instructions="You are a travel recommendation specialist...",
    model=get_model("langdb/travel-recommender") # A virtual model with search tools
)

reply_agent = Agent(
    name="Reply Agent",
    instructions="You reply to the user's query...",
    model=get_model("langdb/reply-formatter") # A virtual model for formatting
)

# Define the orchestrator agent
query_router_agent = Agent(
    name="Query Router",
    instructions="You determine which specialist to use...",
    model=get_model("langdb/query-router"), # A virtual model for routing
    handoffs=[reply_agent],
    tools=[
        booking_agent.as_tool(...),
        travel_recommendation_agent.as_tool(...)
    ]
)
```

### Run the Workflow

Finally, use the `Runner` to execute the workflow. To ensure all steps are linked in the same trace, generate a unique `group_id` for each session and pass it to the `Runner` via `RunConfig`.

```python
# app.py
import asyncio
import uuid

async def run_travel_agent(query: str):
    # A unique group_id links all steps in this session's trace
    group_id = str(uuid.uuid4())

    response = await Runner.run(
        query_router_agent,
        input=query,
        run_config=RunConfig(
            group_id=group_id
        )
    )
    print(response.final_output)

if __name__ == "__main__":
    asyncio.run(run_travel_agent("I want to book a flight to Paris."))
```

---

## 4. Configuring Tools and Guardrails in LangDB

To empower agents with tools or enforce behaviors with guardrails, you use **LangDB Virtual Models**.

1.  In the LangDB UI, navigate to **Models → + New Virtual Model**.
2.  Create virtual models for your agents (e.g., `travel-recommender`, `query-router`).
3.  Attach tools and guardrails as needed:
    *   For the `travel_recommendation_agent`: Attach an **MCP Server** (like Tavily Search) to give it live web search capabilities.
    *   For the `query_router_agent`: Attach **Guardrails** to validate incoming requests (e.g., Topic Adherence, OpenAI Moderation).
4.  Use the virtual model's identifier (e.g., `langdb/travel-recommender`) as the `model` string in your `Agent` definition.

---

## 📚 References

*   [LangDB Docs: Building Travel Agent with OpenAI Agents SDK](https://docs.langdb.ai/guides/building-agents/building-travel-agent-with-openai-agents-sdk)
*   [LangDB Docs: Virtual Models](https://docs.langdb.ai/concepts/virtual-models)
*   [OpenAI Agents SDK on GitHub](https://github.com/openai/agents-sdk)
