"""RAG-grounded reply drafting logic.

Uses the LLM provider adapter to generate a response based on 
the thread history and retrieved FAISS examples.
"""
from __future__ import annotations

from src.llm.provider_adapter import LLMProvider

def draft_reply(
    customer_message: str, 
    thread_history_str: str, 
    retrievals: list[str], 
    intent: str,
    llm: LLMProvider
) -> str:
    """
    RAG-grounded reply drafting. Uses the LLM provider adapter.
    """
    system_prompt = (
        f"You are a helpful customer support agent handling a '{intent}' issue.\n"
        "Draft a professional, empathetic reply to the customer's latest message.\n"
        "Use the provided historical 'Retrieval Examples' to guide your tone and policy answers.\n"
        "Do NOT invent policies. If the examples do not cover the question, state that you will "
        "have a specialist look into it."
    )
    
    retrievals_text = "\n\n".join(retrievals) if retrievals else "None available."
    
    prompt = (
        f"THREAD HISTORY:\n{thread_history_str}\n\n"
        f"HISTORICAL RETRIEVAL EXAMPLES (For reference):\n{retrievals_text}\n\n"
        f"CUSTOMER'S LATEST MESSAGE:\n{customer_message}\n\n"
        f"Please write the final reply to the customer:"
    )
    
    try:
        response = llm.complete(prompt, system_prompt, model_tier='agent')
        if isinstance(response, str):
            return response.strip()
        return "Sorry, I could not generate a response."
    except Exception as e:
        return f"Error drafting reply: {str(e)}"
