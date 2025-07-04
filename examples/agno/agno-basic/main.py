import os
from pylangdb.agno import init
init()

from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.langdb import LangDB

# Configure LangDB-backed model
langdb_model = LangDB(
    id="openai/gpt-4",
    api_key=os.getenv("LANGDB_API_KEY"),
    project_id=os.getenv("LANGDB_PROJECT_ID"),
)

# Create and run your agent
agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=langdb_model,
    tools=[DuckDuckGoTools()],
    instructions="Answer questions using web search",
)

response = agent.run("What is LangDB?")
print(response)