import os
from uuid import uuid4
import asyncio
from agents import (
    Agent,
    Runner,
    function_tool,
    set_default_openai_key,
    set_tracing_disabled,
    set_default_openai_api,
    set_default_openai_client
)
from openai import AsyncOpenAI
from colorama import Fore, Style

# Set default OpenAI API to use only chat_completions
set_default_openai_api(api="chat_completions")
set_tracing_disabled(True)

# Initialize with explicit parameters from environment variables
client = AsyncOpenAI(
    api_key=os.environ.get("LANGDB_API_KEY"),
    base_url=os.environ.get("LANGDB_BASE_URL"),
    default_headers={
        "x-thread-id": str(uuid4()),
        "x-run-id": str(uuid4()),
    }
)
set_default_openai_client(client, use_for_tracing=False)

predefined_queries = [
    "What are your hours of operation?", # Should trigger FAQAgent
    "Please check the status of my order. The ID is 12345.", # Should trigger OrderAgent
    "The food arrived cold and my order was incomplete. I'm very disappointed.", # Should trigger ComplaintAgent
    "I'd like to make a reservation for 4 people tomorrow night.", # Should trigger ReservationAgent
]

# Define different models for different agents
order_agent_model = "anthropic/claude-3.7-sonnet"
faq_agent_model = "gemini/gemini-2.5-pro-exp-03-25"
complaint_agent_model = "xai/grok-2"
reservation_agent_model = "openai/gpt-4o"
classifier_model = "openai/gpt-4o-mini"  

@function_tool
def check_order_status(order_id: str):
    """Check the status of an order with the given order ID."""
    order_statuses = {
        "12345": "Your order 12345 is being prepared and will be delivered in 20 minutes.",
        "67890": "Your order 67890 has been dispatched and will arrive in 10 minutes.",
        "11121": "Your order 11121 is still being processed. Please wait a little longer."
    }
    return order_statuses.get(order_id, "Order ID not found. Please check and try again.")

order_agent = Agent(
    name="OrderAgent",
    model=order_agent_model,
    instructions="Help customers with their order status. If they provide an order ID, fetch the status.",
    tools=[check_order_status]
)


## FAQ Agent


@function_tool
def answer_faq(question: str):
    """Ensure the input is either hours, menu, location, contact, reservation, delivery or allergies """

    faq_responses = {
        "hours": "We are open from 10 AM to 11 PM every day.",
        "menu": "You can find our menu at restaurant.com/menu.",
        "location": "We are located at 123 Main Street, Cityville.",
        "contact": "You can reach us at 555-1234 or email support@restaurant.com.",
        "reservation": "We accept reservations online at restaurant.com/reservations or by calling 555-1234.",
        "delivery": "We offer delivery through our website and on major food delivery platforms like Uber Eats and DoorDash.",
        "allergies": "We accommodate allergies! Please let us know your dietary restrictions when placing an order."
    }
    return faq_responses.get(question.lower(), "I'm not sure, but you can call our helpline at 555-1234.")

faq_agent = Agent(
    name="FAQAgent",
    model=faq_agent_model,
    instructions="Answer common customer questions about hours, menu, and location.\
     Augment the answer based on the tone and details requested in the query \
     Pick up the relevant keyword from the user's query and pass that as input. \
     Example: If user is asking about time then the input keyword is hours.",
    tools=[answer_faq]
)
     


@function_tool
def handle_complaint(complaint: str):
    """Handle customer complaints and ensure respectful communication."""
    return "Thank you for your feedback. We take complaints seriously and will address your concern as soon as possible."

complaint_agent = Agent(
    name="ComplaintAgent",
    model=complaint_agent_model,
    instructions="Handle customer complaints with empathy and ensure respectful communication.",
    tools=[handle_complaint]
)



@function_tool
def handle_reservation(request: str):
    """Ensure the input is either make, modify, cancel or availability """
    reservation_responses = {
        "make": "Your reservation request has been received. Please check your email for confirmation.",
        "modify": "Your reservation modification request has been received. Please check your email for updates.",
        "cancel": "Your reservation has been canceled. We hope to see you another time!",
        "availability": "We have availability for dinner slots from 6 PM to 9 PM. Please book online or call us."
    }
    return reservation_responses.get(request.lower(), "I'm not sure about that request. Please call us at 555-1234 for assistance.")

reservation_agent = Agent(
    name="ReservationAgent",
    model=reservation_agent_model,
    instructions="Assist customers with making, modifying, or canceling reservations.\
     Pick up the relevant keyword from the user's query and pass that as input. \
     Example: If user is asking making a reservation then input keyword is make.",
    tools=[handle_reservation]
)



classifier_agent = Agent(
    name="User Interface Agent",
    model=classifier_model,
    instructions="You are a restaurant customer support agent. \
     Analyze the user query and handoff to ALL appropriate specialized agents. A single query may need responses from multiple agents. \
     - OrderAgent: For questions about order status, order tracking, delivery timing \
     - FAQAgent: For general information about hours, menu, location, allergies, etc. \
     - ComplaintAgent: For handling customer dissatisfaction, complaints, or negative feedback \
     - ReservationAgent: For managing table bookings, reservation changes, or availability checks \
     IMPORTANT: When a query contains multiple topics, make sure to route it to ALL relevant agents.",
    handoffs=[order_agent,faq_agent,complaint_agent,reservation_agent]
)


async def chat(use_predefined=False):
    print("Welcome to the Restaurant Customer Support chat! Type 'exit' to end the chat.")
    response = ""
    
    if use_predefined:
        print("Running predefined queries in conversation flow...\n")
        for i, query in enumerate(predefined_queries, start=1):
            print(Fore.GREEN + f"Query {i}: {query}" + Style.RESET_ALL)
            
            if response:
                input_with_context = response.to_input_list() + [
                    {"role": "user", "content": query}
                ]
            else:
                input_with_context = [{"role": "user", "content": query}]
                
            response = await Runner.run(classifier_agent, input=input_with_context)
            print(Fore.BLUE + f"Support Agent: {response.final_output}" + Style.RESET_ALL)
            
            print(f"{'-'*60}\n")
        return
    
    while True:
        user_input = input(Fore.GREEN + "You: " + Style.RESET_ALL)
        if user_input.lower() == "exit":
            print(Fore.RED + "Goodbye!" + Style.RESET_ALL)
            break

        if response:
          input_with_context = response.to_input_list() + [
          {"role": "user", "content": user_input}
          ]
        else:
          input_with_context = [{"role": "user", "content": user_input}]
        response = await Runner.run(classifier_agent, input=input_with_context)
        print(Fore.BLUE + f"Support Agent: {response.final_output}" + Style.RESET_ALL)


if __name__ == "__main__":
    # Run predefined queries in a flow by default
    asyncio.run(chat(use_predefined=True))
