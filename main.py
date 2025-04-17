from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools.base_tool import Tool
from crewai.tools.structured_tool import CrewStructuredTool
from langchain_community.utilities.duckduckgo_search import \
    DuckDuckGoSearchAPIWrapper

# Define the LLM and search wrapper
try:
    llm = LLM(model="groq/meta-llama/llama-4-scout-17b-16e-instruct")
except Exception as e:
    print(f"Error: {str(e)}")
    llm = LLM(model="openai/gpt-4o")
wrapper = DuckDuckGoSearchAPIWrapper()


def _ddg_search(query: str) -> str:
    """A wrapper around DuckDuckGo Search."""
    try:
        return wrapper.run(query)
    except Exception as e:
        return f"Error: {str(e)}"


crew_search_tool = CrewStructuredTool.from_function(
    _ddg_search,
    name="duckduckgo_search",
    description="A wrapper around DuckDuckGo Search.",
)

search_tool = Tool.from_langchain(crew_search_tool)

# Define the agents and tasks
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
    # tools=[search_tool],
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

# Create the crew and kick off the process
crew = Crew(
    agents=[researcher, reporter],
    tasks=[research_task, report_task],
    process=Process.sequential,
)

try:
    result = crew.kickoff()
    print(result)
except Exception as e:
    print(f"Error: {str(e)}")
