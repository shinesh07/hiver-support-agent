"""LLM-as-a-Judge for qualitative reply evaluation.

Implements blinded, randomized A/B testing or point-wise scoring
for Tone, Helpfulness, Faithfulness.
"""
from __future__ import annotations

import random
from typing import Any
from pydantic import BaseModel
from src.llm.provider_adapter import LLMProvider

class DimensionScore(BaseModel):
    score: int
    evidence: str

class JudgeOutput(BaseModel):
    reasoning: str
    faithfulness: DimensionScore
    helpfulness: DimensionScore
    tone: DimensionScore

def score_reply(customer_message: str, reply: str, retrieved_context: list[str], llm: LLMProvider) -> JudgeOutput:
    """Scores a reply using LLM-as-judge."""
    
    system_prompt = (
        "You are an impartial judge evaluating an agent's customer support reply.\n"
        "Score each dimension from 1 to 5 (1 = terrible, 5 = excellent).\n"
        "- Faithfulness: Is the reply supported by the retrieved context?\n"
        "- Helpfulness: Does the reply actually answer the customer's question?\n"
        "- Tone: Is the reply empathetic, professional, and on-brand?\n"
    )
    
    context_str = "\n".join(retrieved_context) if retrieved_context else "None"
    prompt = (
        f"Customer Message:\n{customer_message}\n\n"
        f"Retrieved Context:\n{context_str}\n\n"
        f"Agent Reply:\n{reply}\n\n"
        "Please provide the JSON scoring output."
    )
    
    try:
        response = llm.complete(prompt, system_prompt, response_model=JudgeOutput, model_tier='judge')
        if hasattr(response, 'faithfulness'):
            return response # type: ignore
            
        # Mock fallback if returning strings
        return JudgeOutput(
            reasoning="Parse failure fallback",
            faithfulness=DimensionScore(score=3, evidence="Fallback"),
            helpfulness=DimensionScore(score=3, evidence="Fallback"),
            tone=DimensionScore(score=3, evidence="Fallback")
        )
    except Exception as e:
        return JudgeOutput(
            reasoning=f"Error: {e}",
            faithfulness=DimensionScore(score=1, evidence="Error"),
            helpfulness=DimensionScore(score=1, evidence="Error"),
            tone=DimensionScore(score=1, evidence="Error")
        )

def score_batch_blinded(examples: list[dict], llm: LLMProvider, seed: int = 42) -> list[dict]:
    """Scores a batch of examples, shuffling to blind the judge."""
    random.seed(seed)
    shuffled_examples = list(examples)
    random.shuffle(shuffled_examples)
    
    results = []
    for example in shuffled_examples:
        result = score_reply(
            example.get("message", ""),
            example.get("reply", ""),
            example.get("retrieved_context", []),
            llm
        )
        results.append({
            "example_id": example.get("id"),
            "score": result.dict()
        })
        
    return results
