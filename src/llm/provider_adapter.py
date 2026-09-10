from __future__ import annotations

import os
from pydantic import BaseModel

class LLMProvider:
    """LLM provider abstraction."""
    def __init__(self, mock_mode: bool = False):
        """Reads OPENAI_API_KEY, OPENAI_MODEL_AGENT, OPENAI_MODEL_JUDGE from env."""
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model_agent = os.getenv("OPENAI_MODEL_AGENT", "gpt-3.5-turbo")
        self.model_judge = os.getenv("OPENAI_MODEL_JUDGE", "gpt-4")
        self.mock_mode = mock_mode
        self.call_count = 0
        self.token_usage = 0
        
    def complete(self, prompt: str, system: str, response_model: type | None = None, model_tier: str = 'agent') -> str | BaseModel:
        """
        model_tier='agent' uses cheaper model, model_tier='judge' uses stronger model.
        Has a mock mode for testing (returns deterministic responses).
        Tracks call count and token usage.
        """
        self.call_count += 1
        self.token_usage += len(prompt) + len(system) # Mock token count
        
        if self.mock_mode:
            if response_model:
                return response_model() # type: ignore
            return "Mock response"
            
        # Real implementation would call OpenAI API
        return "Actual response"
