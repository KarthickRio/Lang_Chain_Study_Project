# Restaurant Customer Support System

A multi-agent system for handling restaurant customer support inquiries using various AI models.

## Prerequisites

- Python 3.8+
- Required packages:
  - openai
  - agents
  - colorama
  - asyncio


## Environment Variables

Create a `.env` file in the project root with the following variables:

```
LANGDB_API_KEY=your_api_key_here
LANGDB_BASE_URL=your_base_url_here
```

## How to Run

1. Install dependencies:
   ```
   pip install colorama openai openai-agents
   ```

2. Run the application:
   ```
   python main.py
   ```
   
   By default, the application will run with predefined queries. To use interactive mode, modify the `use_predefined` parameter to `False` in the main function.
