"""Automated evaluation metrics.

Calculates intent accuracy, escalation recall/precision, and cost.
"""
from __future__ import annotations
import math

def compute_intent_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Compute standard classification metrics for intent."""
    if not y_true:
        return {'accuracy': 0.0}
        
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return {'accuracy': correct / len(y_true)}

def compute_escalation_metrics(
    true_escalate: list[bool],
    pred_escalate: list[bool],
    grounding_available: list[bool]
) -> dict:
    """
    Computes Escalation metrics.
    Fixes Issue #2: We explicitly track Recall on ungrounded cases 
    (where we MUST escalate) and Precision on the full set.
    """
    if not true_escalate:
        return {}
        
    # Recall on ungrounded cases (Critical Safety Metric)
    # y_true = NOT grounded
    n_ungrounded = sum(1 for g in grounding_available if not g)
    tp_ungrounded = sum(1 for i, g in enumerate(grounding_available) if not g and pred_escalate[i])
    
    recall_ungrounded = tp_ungrounded / n_ungrounded if n_ungrounded > 0 else 1.0
    
    # Overall Escalation Rate
    escalation_rate = sum(pred_escalate) / len(pred_escalate)
    
    # Overall Precision: Of the things we escalated, how many ACTUALLY needed it?
    tp_all = sum(1 for t, p in zip(true_escalate, pred_escalate) if t and p)
    precision = tp_all / sum(pred_escalate) if sum(pred_escalate) > 0 else 0.0
    
    # Overall Recall
    recall_all = tp_all / sum(true_escalate) if sum(true_escalate) > 0 else 1.0
    
    return {
        'recall_ungrounded': recall_ungrounded,
        'overall_precision': precision,
        'overall_recall': recall_all,
        'escalation_rate': escalation_rate,
        'f1': 2 * (precision * recall_all) / (precision + recall_all) if (precision + recall_all) > 0 else 0.0
    }

def compute_latency_cost(
    agent_calls: int,
    judge_calls: int,
    avg_tokens_per_call: int = 1500
) -> dict:
    """
    Estimates latency and cost based on LLM calls.
    Mock values for demonstration.
    """
    # E.g. $0.50 per 1M tokens for Agent, $10 per 1M for Judge
    agent_cost = (agent_calls * avg_tokens_per_call / 1_000_000) * 0.50
    judge_cost = (judge_calls * avg_tokens_per_call / 1_000_000) * 10.0
    
    return {
        'total_cost': agent_cost + judge_cost,
        'agent_cost': agent_cost,
        'judge_cost': judge_cost,
        'total_llm_calls': agent_calls + judge_calls
    }
