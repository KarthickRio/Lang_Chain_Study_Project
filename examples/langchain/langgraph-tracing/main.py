# Initialize LangDB tracing
from pylangdb.langchain import init
init()

import os
from typing import Annotated, Sequence, TypedDict
from datetime import datetime

# Import required libraries
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from geopy.geocoders import Nominatim
from pydantic import BaseModel, Field
import requests

# Define the agent state
class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    number_of_steps: int

# Define the weather tool
class SearchInput(BaseModel):
    location: str = Field(description="The city and state, e.g., San Francisco")
    date: str = Field(description="The forecasting date in format YYYY-MM-DD")

@tool("get_weather_forecast", args_schema=SearchInput, return_direct=True)
def get_weather_forecast(location: str, date: str) -> dict:
    """
    Retrieves the weather using Open-Meteo API for a given location (city) and a date (yyyy-mm-dd).
    Returns a dictionary with the time and temperature for each hour.
    """
    geolocator = Nominatim(user_agent="weather-app")
    location = geolocator.geocode(location)
    if not location:
        return {"error": "Location not found"}
    try:
        response = requests.get(
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={location.latitude}&"
            f"longitude={location.longitude}&"
            "hourly=temperature_2m&"
            f"start_date={date}&end_date={date}",
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return {
            time: f"{temp}°C" 
            for time, temp in zip(
                data["hourly"]["time"], 
                data["hourly"]["temperature_2m"]
            )
        }
    except Exception as e:
        return {"error": f"Failed to fetch weather data: {str(e)}"}

# Initialize the model
def create_model():
    """Create and return the ChatOpenAI model with tools bound."""
    api_base = os.getenv("LANGDB_API_BASE_URL")
    api_key = os.getenv("LANGDB_API_KEY")
    project_id = os.getenv("LANGDB_PROJECT_ID")
    default_headers = {
        "x-project-id": project_id,
    }
    llm = ChatOpenAI(
        model_name='openai/gpt-4o', # Choose any model from LangDB
        temperature=0.3,
        openai_api_base=api_base,
        openai_api_key=api_key,
        default_headers=default_headers
    )
    return llm.bind_tools([get_weather_forecast])

# Define the nodes
def call_model(state: AgentState) -> dict:
    """Call the model with the current state and return the response."""
    model = create_model()
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response], "number_of_steps": state["number_of_steps"] + 1}

def route_to_tool(state: AgentState) -> str:
    """Determine the next step based on the model's response."""
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "call_tool"
    return END

# Create the graph
def create_agent():
    """Create and return the LangGraph agent."""
    # Create the graph
    workflow = StateGraph(AgentState)
    workflow.add_node("call_model", call_model)
    workflow.add_node("call_tool", ToolNode([get_weather_forecast]))
    workflow.set_entry_point("call_model")    
    workflow.add_conditional_edges(
        "call_model",
        route_to_tool,
        {
            "call_tool": "call_tool",
            END: END
        }
    )
    workflow.add_edge("call_tool", "call_model")
    return workflow.compile()

def main():
    agent = create_agent()
    query = f"What's the weather in Paris today? Today is {datetime.now().strftime('%Y-%m-%d')}."
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "number_of_steps": 0
    }
    print(f"Query: {query}")
    print("\nRunning agent...\n")
    for output in agent.stream(initial_state):
        for key, value in output.items():
            if key == "__end__":
                continue
            print(f"\n--- {key.upper()} ---")
            if key == "messages":
                for msg in value:
                    if hasattr(msg, 'content'):
                        print(f"{msg.type}: {msg.content}")
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        print(f"Tool Calls: {msg.tool_calls}")
            else:
                print(value)

if __name__ == "__main__":
    main()
