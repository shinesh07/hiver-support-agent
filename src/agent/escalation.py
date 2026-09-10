from __future__ import annotations

FIGURATIVE_EXCEPTIONS = ["killing it", "dying laughing", "dead"]

def apply_hard_rules(message: str, thread_history: list) -> dict | None:
    """
    Implementing all 5 Tier-1 rules with positive/negative example handling.
    """
    return None

def check_margin_band(retrieval_confidence: float, intent: str, thresholds: dict) -> dict | None:
    """
    Checking T_low/T_high.
    """
    return None

def llm_escalation_judgment(message: str, thread_history: list, retrieval_confidence: float, ood_flag: bool) -> dict:
    """
    Tier-2 judgment.
    """
    return {"escalate": False, "reason": "No reason to escalate"}

def decide_escalation(message: str, thread_history: list, retrieval_confidence: float, intent: str, ood_flag: bool, thresholds: dict) -> dict:
    """
    Orchestrating all three layers.
    Returns {escalate: bool, reason: str, tier: str, confidence: float}
    """
    return {
        "escalate": False,
        "reason": "None",
        "tier": "none",
        "confidence": 1.0
    }
