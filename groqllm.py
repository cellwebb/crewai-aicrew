import os

from crewai.llms.base_llm import BaseLLM
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class GroqLLM(BaseLLM):
    """Groq LLM wrapper."""

    def __init__(self, model):
        super().__init__(model=model)
        self.client = client
        self.temperature = 0.7
        self.timeout = 120
        self.max_tokens = 4000
        self.top_p = 0.9
        self.frequency_penalty = 0.1
        self.presence_penalty = 0.1
        self.response_format = {"type": "json"}
        self.seed = 42

    def call(self, messages, tools=None, callbacks=None, available_functions=None):
        """Override call to use Groq client for completions."""
        if isinstance(messages, list):
            prompt = messages[-1]["content"]
        else:
            prompt = messages
        return self.generate(prompt)

    def generate(self, prompt):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            logprobs=self.logprobs,
        )
        return completion.choices[0].message.content
