from openai import OpenAI
from dotenv import load_dotenv
from supabase import create_client, Client
import os

load_dotenv()
api_key = os.getenv("LANGDB_API_KEY")
project_id = os.getenv("LANGDB_PROJECT_ID")
base_url = f"https://api.us-east-1.langdb.ai/{project_id}/v1"


supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_API_KEY")
supabase: Client = create_client(supabase_url, supabase_key)


client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

text = "Hello LangDB"
response = client.embeddings.create(
    model="text-embedding-ada-002",
    input=text,
)

embedding = response.data[0].embedding

# # Store in Supabase
result = supabase.table('embeddings').insert({
    "content": text,
    "embedding": embedding
}).execute()

