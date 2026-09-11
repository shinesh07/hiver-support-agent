"""Context Budget Manager.

Enforces the 5100 token context window allocation:
- System Prompt: 800
- Thread History: 1500
- Retrievals (RAG): 2000
- Generation: 800

Includes fix for Issue #13: Truncate large retrieval examples rather than dropping them.
"""
from __future__ import annotations

from dataclasses import dataclass

@dataclass
class ContextBudget:
    system_limit: int = 800
    thread_limit: int = 1500
    retrieval_limit: int = 2000
    generation_limit: int = 800
    
    @property
    def total_limit(self) -> int:
        return self.system_limit + self.thread_limit + self.retrieval_limit + self.generation_limit


import tiktoken

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Exact token counting to prevent context overflow (Fix for CJK/Emoji bombs)."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback if tiktoken fails
        return len(text) // 4


def truncate_thread(turns: list[dict], budget: int = 1500) -> str:
    """Format thread history, truncating older messages if over budget."""
    formatted = ""
    current_tokens = 0
    
    # Process from newest to oldest
    for turn in reversed(turns):
        role = turn['role'].capitalize()
        text = turn['text']
        
        turn_str = f"{role}: {text}\n"
        turn_tokens = count_tokens(turn_str)
        
        if current_tokens + turn_tokens > budget:
            # If even the single newest message is too big, truncate it
            if current_tokens == 0:
                allowed_chars = budget * 4
                trunc_text = turn_str[:allowed_chars-10] + "... [TRUNC]\n"
                formatted = trunc_text
            else:
                formatted = "[...older messages truncated...]\n" + formatted
            break
            
        formatted = turn_str + formatted
        current_tokens += turn_tokens
        
    return formatted.strip()


def truncate_retrievals(retrievals: list[dict], budget: int = 2000) -> list[str]:
    """Format retrieval examples, truncating individually if needed (Issue #13 fix)."""
    formatted_sources = []
    
    if not retrievals:
        return formatted_sources
        
    # Budget per retrieval to ensure all get some representation
    per_example_budget = budget // len(retrievals)
    
    for r in retrievals:
        # Convert full thread to string
        thread_str = ""
        for t in r['full_thread']:
            thread_str += f"{t['role'].capitalize()}: {t['text']}\n"
            
        header = f"Example (Intent: {r['intent']}, Similarity: {r['similarity']:.3f}):\n"
        full_str = header + thread_str
        
        tokens = count_tokens(full_str)
        if tokens > per_example_budget:
            # Issue #13 fix: Truncate instead of dropping
            # Prevent negative slicing if budget is extremely small
            allowed_chars = max(25, per_example_budget * 4)
            full_str = full_str[:allowed_chars-20] + "... [TRUNCATED]\n"
            
        formatted_sources.append(full_str.strip())
        
    return formatted_sources
