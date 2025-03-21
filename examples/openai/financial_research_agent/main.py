import asyncio
import os
from uuid import uuid4
from openai import AsyncOpenAI
from agents import (
    set_default_openai_key,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled
)

from manager import FinancialResearchManager

# Configure OpenAI settings
set_tracing_disabled(True)
api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    set_default_openai_key(key=api_key, use_for_tracing=False)
    
# Set default OpenAI API to use only chat_completions
set_default_openai_api(api="chat_completions")

# Initialize with explicit parameters from environment variables
client = AsyncOpenAI(
    api_key=api_key,
    base_url=os.environ.get("OPENAI_BASE_URL"),
    default_headers = {
        "x-thread-id": str(uuid4()),
        "x-run-id": str(uuid4())
        # "x-tags": "financial-research"
    }
)

# Set the client for tracing
set_default_openai_client(client, use_for_tracing=True)


# Entrypoint for the financial bot example.
# Run this as `python -m examples.financial_bot.main` and enter a
# financial research query, for example:
# "Write up an analysis of Apple Inc.'s most recent quarter."
async def main() -> None:
    query = input("Enter a financial research query: ")
    mgr = FinancialResearchManager()
    await mgr.run(query)


if __name__ == "__main__":
    asyncio.run(main())
