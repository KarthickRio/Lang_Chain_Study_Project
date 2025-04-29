from mem0 import Memory
from dotenv import load_dotenv


load_dotenv()
import os

langdb_api_key = os.getenv("LANGDB_API_KEY")
langdb_project_id = os.getenv("LANGDB_PROJECT_ID")
base_url =  f"https://api.us-east-1.langdb.ai/{langdb_project_id}/v1"  


config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o",
            "temperature": 0.0,
            "api_key": langdb_api_key,
            "openai_base_url": base_url,
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-ada-002",
            "api_key": langdb_api_key,
            "openai_base_url": base_url,
        },
    }
}




m = Memory.from_config(config_dict=config)

result = m.add(
    "I like to take long walks on weekends.",
    user_id="alice",
    metadata={"category": "hobbies"},
)

print(result)