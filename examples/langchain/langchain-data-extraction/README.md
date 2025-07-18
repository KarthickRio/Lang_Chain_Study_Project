# LangGraph Data Extraction with LangDB

This example demonstrates how to build a complex LangGraph agent for extracting structured information from meeting transcripts using LangDB for tracing and observability. The agent implements advanced workflow patterns including multi-stage processing, validation, and iterative refinement.

This sample showcases LangDB's comprehensive tracing capabilities with LangGraph, providing full observability into complex agent workflows.

## Overview

This sample builds a sophisticated LangGraph agent that processes meeting transcripts and extracts structured, meaningful information. It implements a multi-stage workflow with preprocessing, extraction, validation, refinement, and synthesis phases. The agent includes confidence scoring, conditional routing, fallback mechanisms, and robust error handling. All execution is traced through LangDB for complete observability into the agent's decision-making process and performance.

## Getting Started

First, clone the repository and navigate to this example:

```bash
git clone https://github.com/langdb/langdb-samples.git
cd langdb-samples/examples/langchain/langchain-data-extraction
```

## Prerequisites

1. **LangDB Account**: Ensure you have a LangDB account and have created a project to get your `LANGDB_API_KEY` and `LANGDB_PROJECT_ID`.
2. **Python Environment**: It is recommended to use a virtual environment.

## 1. Setup

Create a `.env` file in this directory (`examples/langchain/langchain-data-extraction`) and add your credentials:

```sh
# examples/langchain/langchain-data-extraction/.env

LANGDB_API_KEY="<your_langdb_api_key>"
LANGDB_PROJECT_ID="<your_langdb_project_id>"
LANGDB_API_BASE_URL='https://api.us-east-1.langdb.ai'
```

## 2. Installation

Install the required Python packages:

```sh
pip install -r requirements.txt
```

## 3. How It Works

This example demonstrates a sophisticated LangGraph agent with the following key features:

### Structural Tracing with `init()`

The `main.py` script calls `init()` from `pylangdb.langchain` at the very beginning. This instruments the LangChain and LangGraph libraries to capture the high-level structure of your graph, including nodes, edges, and tool calls.

```python
from pylangdb.langchain import init
init()
```

### Multi-Stage Processing Workflow

The agent implements a complex workflow with multiple stages:

1. **Preprocessing**: Analyzes transcript structure and determines complexity
2. **Initial Extraction**: Extracts comprehensive information with confidence scoring
3. **Validation**: Validates extraction quality and provides feedback
4. **Refinement**: Refines extraction based on validation feedback
5. **Synthesis**: Produces final comprehensive summary

### Modular Architecture

The project is organized into modular components for better maintainability:

- **`main.py`** - Main execution script with minimal logic
- **`models.py`** - Data models, enums, and state definitions
- **`tools.py`** - Tool definitions for extraction and validation
- **`nodes.py`** - Node functions and routing logic
- **`agent.py`** - Agent construction and workflow definition
- **`transcript.py`** - Sample transcript data for testing

### Advanced Features

- **Confidence Scoring**: Each extraction section gets a confidence score
- **Conditional Routing**: Smart routing based on validation results
- **Fallback Mechanisms**: Simplified extraction if complex workflow fails
- **Iterative Refinement**: Multiple attempts with feedback loops
- **Error Handling**: Robust error handling with fallback mechanisms

## Usage

Run the example script:

```sh
python main.py
```

The agent will process a complex meeting transcript and provide detailed output showing:
- Processing phases and their results
- Confidence scores for each extraction section
- Validation feedback and improvements
- Final synthesized summary

After the script completes, you will see the full, detailed trace of the LangGraph execution in your LangDB project dashboard.

## Customization

- Modify `transcript.py` to use your own transcript data
- Adjust confidence thresholds in `tools.py`
- Add new extraction phases in `models.py`
- Extend the workflow in `agent.py`

## Output

The agent provides detailed output showing:
- Processing phases and their results
- Confidence scores for each extraction section
- Validation feedback and improvements
- Final synthesized summary 