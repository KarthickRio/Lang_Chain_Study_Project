#!/usr/bin/env python3

import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
import sys
from pylangdb.crewai import init

load_dotenv()
init()


# Initialize the OpenAI API
api_key = os.environ.get("LANGDB_API_KEY")
api_base = os.environ.get("LANGDB_API_BASE_URL")
project_id = os.environ.get("LANGDB_PROJECT_ID")


# Base LLM configuration
def create_llm(model):
    return LLM(
        model=model,
        api_key=api_key,
        base_url=api_base,
        extra_headers={
            "x-project-id": project_id
        }
    )

@CrewBase
class ReportGenerationCrew():
    """Report Generation crew"""

    agents_config = "configs/agents.yaml"
    tasks_config = "configs/tasks.yaml"

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True,
            llm=create_llm("openai/langdb/reportresearcher_9wzgx5n5")
        )

    @agent
    def analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['analyst'],
            verbose=True,
            llm=create_llm("openai/anthropic/claude-3.7-sonnet")
        )

    @agent
    def report_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['report_writer'],
            verbose=True,
            llm=create_llm("openai/gemini/gemini-2.5-pro-preview")
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
            agent=self.researcher()
        )

    @task
    def analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['analysis_task'],
            agent=self.analyst()
        )

    @task
    def report_writing_task(self) -> Task:
        return Task(
            config=self.tasks_config['report_writing_task'],
            agent=self.report_writer(),
            markdown=True,
            output_file="report.md"
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.researcher(),
                self.analyst(),
                self.report_writer()
            ],
            tasks=[
                self.research_task(),
                self.analysis_task(),
                self.report_writing_task()
            ],
            process=Process.sequential,
            verbose=True
        )

def generate_report(topic):
    if not topic:
        print("Error: Please provide a topic to research and report on.")
        return None
    
    print(f"Starting report generation for topic: {topic}")
    
    # Create crew instance
    crew_instance = ReportGenerationCrew()
    
    # Update task descriptions with the specific topic
    crew_instance.research_task().description = f"""Research the topic: {topic}
    
    Gather comprehensive information including:
    - Current trends and developments
    - Key statistics and metrics
    - Industry insights and expert opinions
    - Recent news and updates
    - Market analysis (if applicable)
    
    Focus on finding credible sources and recent information."""
    
    crew_instance.report_writing_task().description = f"""Create a professional report about {topic} in Markdown format.
    
    The report should include:
    - Executive Summary (key takeaways in 2-3 paragraphs)
    - Introduction and background
    - Key Findings (organized into clear sections)
    - Detailed Analysis
    - Recommendations and next steps
    - Conclusion
    
    Ensure the report is well-formatted, professional, and actionable."""
    
    try:
        result = crew_instance.crew().kickoff()
        
        print(f"\nReport generated successfully!")
        print(f"Report saved as: report.md")
        return result
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return None

def main():
    print("CrewAI Report Writing Agent")
    print("=" * 40)
    
    

    # Get topic from command line arguments or user input
    if len(sys.argv) > 1:
        topic = ' '.join(sys.argv[1:])
    else:
        topic = input("\nEnter the topic you want to research and create a report about: ").strip()
    
    if not topic:
        print("No topic provided. Exiting.")
        return
    
    # Generate the report
    result = generate_report(topic)
    
    if result:
        print("\n" + "=" * 40)
        print("REPORT PREVIEW:")
        print("=" * 40)
        print(str(result)[:500] + "..." if len(str(result)) > 500 else str(result))

if __name__ == "__main__":
    main()