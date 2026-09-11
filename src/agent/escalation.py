"""Escalation routing logic.

Implements the 3-layer escalation defense:
1. Hard rules (regex for severe keywords/OOD)
2. Margin band check (k-NN similarity vs calibrated thresholds)
3. LLM judgment (if falls inside the ambiguous margin band)

Includes support for per-intent thresholds and shrinkage estimators (Issue #5).
"""
from __future__ import annotations

import re
from src.llm.provider_adapter import LLMProvider


FIGURATIVE_EXCEPTIONS = [
    r'kill\s+for\s+a\s+(coffee|drink|beer)',
    r'die\s+for\s+a\s+(coffee|drink|beer)',
    r'(this|it)\s+(is|was)\s+a\s+joke',
    r'just\s+kidding'
]

HARD_ESCALATE_PATTERNS = [
    r'\b(lawsuit|lawyer|attorney|sue|legal action)\b',
    r'\b(fraud|scam|stolen|hacked)\b',
    r'\b(manager|supervisor)\b',
    r'\b(refund now|chargeback)\b',
    r'\b(kill|die|suicide)\b' # Safety
]

def apply_hard_rules(customer_message: str, is_ood: bool = False) -> tuple[bool, str]:
    """Layer 1: Deterministic escalation."""
    msg_lower = customer_message.lower()
    
    # 1. Check figurative exceptions first
    for pattern in FIGURATIVE_EXCEPTIONS:
        if re.search(pattern, msg_lower):
            # Strip it out so it doesn't trigger hard rules
            msg_lower = re.sub(pattern, '', msg_lower)
            
    # 2. Hard escalate patterns
    for pattern in HARD_ESCALATE_PATTERNS:
        if re.search(pattern, msg_lower):
            return True, f"Hard rule triggered: {pattern}"
            
    # 3. OOD detection
    if is_ood:
        return True, "Hard rule triggered: OOD (Out of Distribution)"
        
    return False, ""


def check_margin_band(
    retrieval_similarity: float, 
    intent: str, 
    thresholds: dict, 
    global_threshold: float,
    margin_width: float = 0.05
) -> str:
    """
    Layer 2: Check similarity against the threshold.
    Returns: 'auto_reply', 'escalate', or 'llm_judge'
    """
    # Issue #5 fix: we rely on threshold_sweep.py to provide shrinkage-adjusted thresholds.
    # If the intent isn't in the dict, fall back to global.
    T = thresholds.get(intent, global_threshold)
    
    # Define the margin band [T - margin, T + margin]
    T_high = T + margin_width
    T_low = T - margin_width
    
    if retrieval_similarity >= T_high:
        return 'auto_reply'
    elif retrieval_similarity < T_low:
        return 'escalate'
    else:
        return 'llm_judge'


def llm_escalation_judgment(
    customer_message: str, 
    thread_history: str, 
    llm: LLMProvider
) -> tuple[bool, str]:
    """Layer 3: Expensive LLM judgment for edge cases in the margin band."""
    
    system_prompt = (
        "You are an escalation judge for a customer support agent. "
        "Review the conversation and decide if it must be escalated to a human. "
        "Escalate IF: \n"
        "- The customer is highly irate, swearing, or making threats.\n"
        "- The issue is highly complex, involves safety, fraud, or legal threats.\n"
        "- The customer explicitly demands a human, manager, or supervisor.\n"
        "Otherwise, do NOT escalate.\n"
        "Reply with a JSON object containing a boolean 'escalate' and a brief 'reason'."
    )
    
    prompt = f"THREAD HISTORY:\n{thread_history}\n\nLATEST MESSAGE:\n{customer_message}"
    
    try:
        response = llm.complete(prompt, system_prompt, model_tier='judge')
        # In mock mode or real mode, parse the JSON
        if isinstance(response, str):
            import json
            import re
            try:
                # Strip markdown code blocks if the LLM adds them
                clean_response = re.sub(r'```json\n|\n```|```', '', response).strip()
                data = json.loads(clean_response)
                return data.get('escalate', True), data.get('reason', 'Parse successful')
            except json.JSONDecodeError:
                pass
        
        # Default to safe if parsing fails
        return True, "LLM parsing failed, defaulting to escalate"
        
    except Exception as e:
        return True, f"LLM call failed: {str(e)}"

def decide_escalation(
    customer_message: str,
    thread_history_str: str,
    retrieval_similarity: float,
    intent: str,
    thresholds: dict,
    global_threshold: float,
    is_ood: bool,
    margin_width: float,
    llm: LLMProvider
) -> tuple[bool, str, str]:
    """
    Run the full 3-layer escalation pipeline.
    Returns: (should_escalate, reason, layer_triggered)
    """
    # Layer 1
    hard_esc, reason = apply_hard_rules(customer_message, is_ood)
    if hard_esc:
        return True, reason, "Layer 1 (Hard Rules)"
        
    # Layer 2
    decision = check_margin_band(retrieval_similarity, intent, thresholds, global_threshold, margin_width)
    
    if decision == 'auto_reply':
        return False, f"Similarity {retrieval_similarity:.3f} > T_high", "Layer 2 (Margin Band)"
    elif decision == 'escalate':
        return True, f"Similarity {retrieval_similarity:.3f} < T_low", "Layer 2 (Margin Band)"
        
    # Layer 3
    llm_esc, reason = llm_escalation_judgment(customer_message, thread_history_str, llm)
    return llm_esc, reason, "Layer 3 (LLM Judge)"
