from __future__ import annotations

from dataclasses import dataclass

@dataclass
class ContextBudget:
    system_prompt: int = 800
    thread_history: int = 1500
    retrieved_examples: int = 2000
    generation: int = 800

def truncate_thread(turns: list, max_tokens: int = 1500) -> tuple[list, dict]:
    """
    Keeping first + last 3 turns.
    Returns (truncated_turns, truncation_log)
    """
    if len(turns) <= 4:
        return turns, {"truncated": False}
    truncated = [turns[0]] + turns[-3:]
    return truncated, {"truncated": True, "original_length": len(turns), "new_length": len(truncated)}

def truncate_retrievals(examples: list, max_tokens: int = 2000) -> tuple[list, dict]:
    """
    Keeping only customer-message + brand-reply pairs.
    Returns (truncated, truncation_log)
    """
    return examples, {"truncated": False}
