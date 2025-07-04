# Initialize LangDB tracing
from pylangdb.openai import init
init()

# Agent SDK imports
from agents import (
    Agent,
    Runner,
    set_default_openai_client,
    set_default_openai_api,
    RunConfig,
    Model,
    OpenAIChatCompletionsModel
)
from openai import AsyncOpenAI
import os
import uuid
import asyncio


# Configure the OpenAI client with LangDB headers
client = AsyncOpenAI(api_key=os.environ["LANGDB_API_KEY"],
        base_url=os.environ["LANGDB_API_BASE_URL"],
        default_headers={"x-project-id": os.environ["LANGDB_PROJECT_ID"]})

# Set the configured client as default with tracing enabled
set_default_openai_client(client, use_for_tracing=True)
set_default_openai_api(api="chat_completions")
# set_default_openai_key(os.environ["LANGDB_API_KEY"])

# Create a custom model provider for advanced routing
def get_model(model_name) -> Model:
    return OpenAIChatCompletionsModel(model=model_name, openai_client=client)

math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
    model=get_model("anthropic/claude-3.7-sonnet") # Choose any model available on LangDB
)

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly.",
    model=get_model("gemini/gemini-2.0-flash") # Choose any model available on LangDB
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent],
    model=get_model("openai/gpt-4o-mini") # Choose any model available on LangDB
)

# Define async function to run the agent
async def run_agent():
    response = await Runner.run(
        triage_agent,
        input="who was the first president of the united states?",
        run_config=RunConfig(
            group_id=str(uuid.uuid4())                 # Link all steps to the same trace
        )
    )
    print(response.final_output)

# Run the async function with asyncio
asyncio.run(run_agent())
