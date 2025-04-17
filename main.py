from crewai import LLM, Agent, Crew, Process, Task
from langchain.tools import DuckDuckGoSearchRun

llm = LLM(model="groq/meta-llama/llama-4-scout-17b-16e-instruct")
search_tool = DuckDuckGoSearchRun()

researcher = Agent(
    role="Researcher",
    goal="Research the latest AI developments",
    backstory="You are a researcher with a passion for AI",
    llm=llm,
    verbose=True,
    tools=[search_tool],
)
reporter = Agent(
    role="Reporter",
    goal="Report on the latest AI developments",
    backstory="You are a reporter with a passion for AI",
    llm=llm,
    verbose=True,
    tools=[search_tool],
)

research_task = Task(
    description="Research the latest AI developments",
    expected_output="A list of findings on the latest AI developments.",
    agent=researcher,
)
report_task = Task(
    description="Report on the latest AI developments",
    expected_output="A human-readable report summarizing the latest AI developments.",
    agent=reporter,
)

crew = Crew(
    agents=[researcher, reporter],
    tasks=[research_task, report_task],
    process=Process.sequential,
)
result = crew.kickoff()

print(result)
