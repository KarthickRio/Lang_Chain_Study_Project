# Reasoning Finance Team Example

This example demonstrates a multi-agent team for collaborative financial analysis using LangDB and the `agno` library.

This sample is based on the [LangDB documentation for building a reasoning finance team](https://docs.langdb.ai/guides/building-agents/building-a-reasoning-finance-team-with-agno).

## Getting Started

First, clone the repository and navigate to this example:

```bash
git clone https://github.com/langdb/langdb-samples.git
cd langdb-samples/examples/agno/reasoning-finance-team
```

The system is composed of two specialist agents orchestrated by a coordinating team, all powered and monitored by LangDB.

## Overview

The system is composed of two specialist agents orchestrated by a coordinating team:

1. **Web Search Agent**: Responsible for gathering the latest news and market sentiment from the internet. This agent uses a LangDB Virtual Model, which allows its underlying search tool (e.g., Tavily) to be configured dynamically in the LangDB UI without changing the agent's code.
2. **Finance Agent**: Equipped with `YFinanceTools` to fetch and analyze quantitative stock data, including pricing, fundamentals, and analyst recommendations. This agent is powered by the Grok-4 model.
3. **Reasoning Finance Team**: A coordinator that directs the two agents, synthesizes their findings, and produces a final, comprehensive report.

LangDB provides the backbone for this system, enabling seamless model access, tool integration, and full observability into each agent's actions and the team's collaborative process.

## Installation and Setup

1. **Install Dependencies**: Make sure you have Python 3.8+ installed. Then, install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

2. **Set Environment Variables**

   Create a `.env` file in this directory (`examples/agno/reasoning-finance-team`) and add your credentials:

   ```sh
   # examples/agno/reasoning-finance-team/.env

   LANGDB_API_KEY="<your_langdb_api_key>"
   LANGDB_PROJECT_ID="<your_langdb_project_id>"
   ```

   Replace `<your_langdb_api_key>` and `<your_langdb_project_id>` with your actual LangDB credentials.

3. **Configure Virtual Models and Tools (in LangDB UI)**:

    To empower the `web_agent` with live web search capabilities, you need to configure a Virtual Model in the LangDB UI.

    * **Create a Virtual MCP Server**:
        1. In the LangDB UI, navigate to **Projects → MCP Servers**.
        2. Click **+ New Virtual MCP Server** and configure it:
            * **Name**: `web-search-mcp`
            * **Underlying MCP**: Select **Tavily Search**.
            * **Note**: The Tavily MCP requires an API key. Ensure you have added your `TAVILY_API_KEY` to your LangDB account secrets.

    * **Create and Configure the Virtual Model**:
        1. Navigate to **Models → + New Virtual Model**.
        2. Give it a name (e.g., `search-agent`).
        3. In the **Tools** section, click **+ Attach MCP Server** and select the `web-search-mcp` you created.
        4. Save the model and copy its identifier (e.g., `langdb/search-agent_xxxxxx`).
        5. Use this identifier as the model in your `web_agent` definition in `main.py`.

## Code Walkthrough

### 1. Initialize LangDB

The script starts by initializing LangDB to enable automatic tracing and model routing. This is done before importing any other components from the `agno` library.

```python
from pylangdb.agno import init
init()
```

### 2. Define the Web Search Agent

The `web_agent` is responsible for searching the web. Instead of hard-coding a search tool, we assign it a LangDB Virtual Model. This decouples the agent's logic from the specific tools it uses.

```python
web_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=LangDB(id="langdb/search_agent_xmf4v5jk"),
    instructions="Always include sources"
)
```

### 3. Define the Finance Agent

This agent is equipped with `YFinanceTools` to access a wide range of financial data. It is powered by Grok-4 and has specific instructions to format its output professionally.

```python
finance_agent = Agent(
    name="Finance AI Agent",
    role="Analyse the given stock",
    model=LangDB(id="xai/grok-4"),
    tools=[YFinanceTools(...)],
)
```

### 4. Define the Coordinating Team

The `ReasoningFinanceTeam` orchestrates the two agents. It operates in `coordinate` mode, allowing it to delegate tasks, synthesize information, and ensure the final output meets the specified success criteria.

```python
reasoning_finance_team = Team(
    name="Reasoning Finance Team",
    mode="coordinate",
    model=LangDB(id="xai/grok-4"),
    members=[web_agent, finance_agent],
    # ...
)
```

## Running the Example

To run the example, simply execute the `main.py` script:

```bash
python main.py
```

The script will then output a comparative financial analysis of Apple (AAPL), Google (GOOGL), and Microsoft (MSFT).
