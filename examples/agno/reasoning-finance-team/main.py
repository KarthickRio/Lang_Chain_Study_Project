import os
from dotenv import load_dotenv

from pylangdb.agno import init
init()

from agno.agent import Agent
from agno.team.team import Team
from agno.tools.yfinance import YFinanceTools
from agno.models.langdb import LangDB

load_dotenv()

# Web Search Agent with Tavily via LangDB Virtual Model
web_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=LangDB(id="langdb/search_agent_xmf4v5jk"),
    instructions="Always include sources"
)

# Finance Agent powered by Grok 4
finance_agent = Agent(
    name="Finance AI Agent",
    role="Analyse the given stock",
    model=LangDB(id="xai/grok-4"),
    tools=[YFinanceTools(
        stock_price=True,
        stock_fundamentals=True,
        analyst_recommendations=True,
        company_info=True,
        company_news=True
    )],
    instructions=[
        "Use tables to display stock prices, fundamentals (P/E, Market Cap), and recommendations.",
        "Clearly state the company name and ticker symbol.",
        "Focus on delivering actionable financial insights."
    ]
)

# Multi-agent team for collaborative financial analysis
reasoning_finance_team = Team(
    name="Reasoning Finance Team",
    mode="coordinate",
    model=LangDB(id="xai/grok-4"),
    members=[web_agent, finance_agent],
    instructions=[
        "Collaborate to provide comprehensive financial and investment insights",
        "Consider both fundamental analysis and market sentiment",
        "Use tables and charts to display data clearly and professionally",
        "Present findings in a structured, easy-to-follow format",
        "Only output the final consolidated analysis, not individual agent responses"
    ],
    markdown=True,
    show_members_responses=True,
    success_criteria="The team has provided a complete financial analysis with data, visualizations, risk assessment, and actionable investment recommendations supported by quantitative analysis and market research."
)

reasoning_finance_team.print_response(
    """Compare the tech sector giants (AAPL, GOOGL, MSFT) performance:\n    1. Get financial data for all three companies\n    2. Analyze recent news affecting the tech sector\n    3. Calculate comparative metrics and correlations\n    4. Recommend portfolio allocation weights"""
)
