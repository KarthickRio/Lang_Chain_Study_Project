#!/usr/bin/env python3

import os
import sys
from pylangdb.crewai import init
init()
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

load_dotenv()

def create_llm(model):
    return LLM(
        model=model,
        api_key=os.environ.get("LANGDB_API_KEY"),
        base_url=os.environ.get("LANGDB_API_BASE_URL"),
        extra_headers={"x-project-id": os.environ.get("LANGDB_PROJECT_ID")}
    )

class ResearchPlanningCrew:
    def researcher(self) -> Agent:
        return Agent(
            role="Research Specialist",
            goal="Research topics thoroughly",
            backstory="Expert researcher with skills in finding information",
            tools=[SerperDevTool()],
            llm=create_llm("openai/gpt-4o"),
            verbose=True
        )
    
    def planner(self) -> Agent:
        return Agent(
            role="Strategic Planner",
            goal="Create actionable plans based on research",
            backstory="Strategic planner who breaks down complex challenges",
            reasoning=True,
            max_reasoning_attempts=3,
            llm=create_llm("openai/anthropic/claude-3.7-sonnet"),
            verbose=True
        )
    
    def research_task(self) -> Task:
        return Task(
            description="Research the topic thoroughly and compile information",
            agent=self.researcher(),
            expected_output="Comprehensive research report"
        )
    
    def planning_task(self) -> Task:
        return Task(
            description="Create a strategic plan based on research",
            agent=self.planner(),
            expected_output="Strategic execution plan with phases and goals",
            context=[self.research_task()]
        )
    
    def crew(self) -> Crew:
        return Crew(
            agents=[self.researcher(), self.planner()],
            tasks=[self.research_task(), self.planning_task()],
            verbose=True,
            process=Process.sequential
        )

def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else "Artificial Intelligence in Healthcare"
    
    crew_instance = ResearchPlanningCrew()
    
    # Update task descriptions with topic
    crew_instance.research_task().description = f"Research {topic} thoroughly and compile information"
    crew_instance.planning_task().description = f"Create a strategic plan for {topic} based on research"
    
    result = crew_instance.crew().kickoff()
    print(result)

if __name__ == "__main__":
    main()