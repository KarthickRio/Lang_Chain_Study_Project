# LangDB Samples

This repository provides comprehensive examples and integrations for LangDB, demonstrating its capabilities across various frameworks and use cases.

## Quick Start

1. Replace the base URL with LangDB's API endpoint
2. Use your LangDB token as the API key
3. Add your project ID in the extra headers
4. Start making API calls using any supported model

## Examples

### Basic Integration

| Framework | Example | Path |
|-----------|---------|------|
| OpenAI API | Simple Integration | [`examples/basic.py`](examples/basic.py) |

### Framework Integrations

| Framework | Example                           | Path                                                                                   |
|-----------|-----------------------------------|----------------------------------------------------------------------------------------|
| LangChain | Basic Integration                 | [`examples/langchain/langchain-basic`](examples/langchain/langchain-basic)             |
| LangChain | Multi-agent Setup                 | [`examples/langchain/langchain-multi-agent`](examples/langchain/langchain-multi-agent) |
| LangChain | RAG-agent Setup                   | [`examples/langchain/langchain-rag-bot`](examples/langchain/langchain-rag-bot)         |
| CrewAI | Basic Implementation              | [`examples/crewai/crewai-basic`](examples/crewai/crewai-basic)                         |
| CrewAI | Multi-agent Orchestration         | [`examples/crewai/crewai-multi-agent`](examples/crewai/crewai-multi-agent)             |
| CrewAI | Report Writing Agent              | [`examples/crewai/report-writing-agent`](examples/crewai/report-writing-agent)         |
| LlamaIndex | Basic Integration                 | [`examples/llamaindex/llamaindex-basic`](examples/llamaindex/llamaindex-basic)         |
| Google ADK | Web Search Agent                  | [`examples/google-adk/web-search-agent`](examples/google-adk/web-search-agent)         |
| OpenAI Agents SDK | Customer Support Agent            | [`examples/openai/customer-support`](examples/openai/customer-support)                 |
| OpenAI Agents SDK | Travel Agent                      | [`examples/openai/travel-agent`](examples/openai/travel-agent)                         |
| Mem0 | Memory System Integration         | [`examples/mem0`](examples/mem0)                                                       |
| Vercel AI SDK | JavaScript/Node.js Implementation | [`examples/vercel`](examples/vercel)                                                   |
| Supabase | Database Integration              | [`examples/supabase`](examples/supabase)                                               |
| Rasa | Conversational AI Integration     | [`examples/rasa`](examples/rasa)                                                       |


### Feature Examples

| Feature | Example | Path |
|---------|---------|------|
| Routing | Basic Setup | [`examples/routing/routing-basic`](examples/routing/routing-basic) |
| Routing | Multi-agent Setup | [`examples/routing/routing-multi-agent`](examples/routing/routing-multi-agent) |
| Evaluation | Model Evaluation & Cost Analysis | [`examples/evaluation`](examples/evaluation) |

### MCP Examples

| Example | Description | Path |
|---------|-------------|------|
| MCP Support | Model Provider Integration | [`examples/mcp/mcp-support.ipynb`](examples/mcp/mcp-support.ipynb) |
| Cafe Dashboard | Next.js with MCP Integration | [`examples/mcp/cafe-dashboard`](examples/mcp/cafe-dashboard) |
| Server Actions Demo | Next.js Server Actions with MCP | [`examples/mcp/nextjs-server-actions-demo`](examples/mcp/nextjs-server-actions-demo) |
| SvelteKit Integration | SvelteKit MCP Sample | [`examples/mcp/sveltekit-mcp-sample`](examples/mcp/sveltekit-mcp-sample) |

## Key Features

🚀 **High Performance**
- Built in Rust for maximum speed and reliability
- Seamless integration with any framework (Langchain, Vercel AI SDK, CrewAI, etc.)
- Integrate with any MCP servers(https://docs.langdb.ai/ai-gateway/features/mcp-support)

📊 **Enterprise Ready**
- [Comprehensive usage analytics and cost tracking](https://docs.langdb.ai/ai-gateway/features/analytics)
- [Rate limiting and cost control](https://docs.langdb.ai/ai-gateway/features/usage)
- [Advanced routing, load balancing and failover](https://docs.langdb.ai/ai-gateway/features/routing)
- [Evaluations](https://docs.langdb.ai/ai-gateway/features/evaluation)

🔒 **Data Control**
- Full ownership of your LLM usage data
- Detailed logging and tracing

### Looking for More? Try Our Hosted & Enterprise Solutions

🌟 **[Hosted Version](https://langdb.ai)** - Get started in minutes with our fully managed solution
- Zero infrastructure management
- Automatic updates and maintenance
- Pay-as-you-go pricing

💼 **[Enterprise Version](https://langdb.ai/)** - Enhanced features for large-scale deployments
- Advanced team management and access controls
- Custom security guardrails and compliance features
- Intuitive monitoring dashboard
- Priority support and SLA guarantees
- Custom deployment options

[Contact our team](https://calendly.com/d/cqs2-cfz-gdn/meet-langdb-team) to learn more about enterprise solutions.

## Built for Developers

LangDB's AI Gateway is designed with developers in mind, focusing on providing a practical and streamlined experience for integrating LLMs into your workflows. Whether you're building a new AI-powered application or enhancing existing systems, LangDB makes it easier to manage and scale your LLM implementations.

## Support

For more information and support, visit our [documentation](https://docs.langdb.ai).
