import os

from crewai.llm import LLM
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class GroqLLM(LLM):
    """Groq LLM wrapper."""

    def __init__(self, model):
        super().__init__(model=model)
        self.client = client
        self.model = model
        self.response_format = None
        self.is_anthropic = False
        self.timeout = None
        self.top_p = None
        self.temperature = None
        self.n = None
        self.max_tokens = None
        self.max_completion_tokens = None
        self.presence_penalty = None
        self.frequency_penalty = None
        self.logit_bias = None
        self.seed = 42069
        self.logprobs = None
        self.top_logprobs = None
        self.api_base = None
        self.base_url = None

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
