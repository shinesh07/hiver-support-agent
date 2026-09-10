from __future__ import annotations

from src.agent.retrieval_index import RetrievalIndex
from src.agent.intent_classifier import IntentClassifier
from src.agent.ood_detector import OODDetector
from src.agent.draft_reply import draft_reply
from src.agent.escalation import decide_escalation

def classify_and_respond(message: str, thread_history: list, index: RetrievalIndex, classifier: IntentClassifier, ood_detector: OODDetector, thresholds: dict) -> dict:
    """
    Main orchestrator calling stages A->B->C->D in sequence.
    Returns {intent, draft_reply, grounding_sources, retrieval_confidence, escalate, escalate_reason, ood_flag, truncation_log}
    """
    # A. OOD detection
    ood_flag, _ = ood_detector.is_ood(None) # type: ignore
    
    # B. Intent classification
    intent_res = classifier.classify(message, thread_history)
    intent = intent_res.get('intent', 'general')
    
    # C. Draft reply
    draft = draft_reply(message, thread_history, [], intent)
    
    # D. Escalation
    esc = decide_escalation(message, thread_history, draft['retrieval_confidence'], intent, ood_flag, thresholds)
    
    return {
        "intent": intent,
        "draft_reply": draft['reply'],
        "grounding_sources": draft['grounding_sources'],
        "retrieval_confidence": draft['retrieval_confidence'],
        "escalate": esc['escalate'],
        "escalate_reason": esc['reason'],
        "ood_flag": ood_flag,
        "truncation_log": {}
    }
