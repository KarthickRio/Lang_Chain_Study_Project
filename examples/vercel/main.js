import { generateText } from "ai"
import { createLangDB } from '@langdb/vercel-provider';

// Ensure you have LANGDB_API_KEY set in your environment variables
const langdbApiKey = process.env.LANGDB_API_KEY;
const LANGDB_PROJECT_ID = "1641f1db-bab8-4687-8a0e-efecd95a5361";

if (!langdbApiKey) {
  console.error("Error: LANGDB_API_KEY environment variable is not set.");
  process.exit(1);
}

const langdb = createLangDB({
  apiKey: langdbApiKey, 
  projectId: LANGDB_PROJECT_ID,
});

const { text } = await generateText({
  model: langdb('openai/gpt-4o-mini'),
  prompt: "What is Capital of France?"
});
console.log(text);