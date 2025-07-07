# Routing Multi-Agent 

This example demonstrates how to use CrewAI to build a multi-agent system that routes questions to the most relevant agent.

## Getting Started

First, clone the repository and navigate to this example:

```bash
git clone https://github.com/langdb/langdb-samples.git
cd langdb-samples/examples/routing/routing-multi-agent
```

## Pre-requisites

```bash 
pip install -U langchain_community crewai langchain_experimental matplotlib pandas crewai[tools] tqdm openai   
```

## API-Key

Put in your .env file
```bash
API_KEY="LANGDB_API_KEY"           # LangDB API Key
LANGDB_API_KEY="$API_KEY"           
OPENAI_API_KEY="$API_KEY"           
OPENAI_API_BASE="https://us-east-1.langdb.ai"
```
