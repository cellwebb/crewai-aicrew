from crewai import Agent, Crew, Task

stock_researcher = Agent(
    role="stock_researcher",
    goal="Analyze the stock market and provide insights.",
    backstory="Experienced stock researcher with a deep understanding of the stock market and its complexities.",
    verbose=True,
)

stock_analyzer = Agent(
    role="stock_analyzer",
    goal="Interpret stock data and provide recommendations.",
    backstory="Skilled stock analyzer with a strong understanding of market trends and indicators.",
    verbose=True,
)

research = Task(
    description="Research stock data for the given ticker.",
    expected_output="A list of dictionaries containing stock data (e.g., date, open, close, volume).",
    agent=stock_researcher,
)

analyze = Task(
    description="Analyze the researched stock data and make recommendations.",
    expected_output="A list of recommendations based on the analyzed stock data.",
    agent=stock_analyzer,
)

crew = Crew(
    agents=[stock_researcher, stock_analyzer],
    tasks=[research, analyze],
)

result = crew.kickoff(inputs={"ticker": "AAPL"})

print(result)
