"""Trivial Baseline.

Uses zero machine learning:
- Intent: Always predicts the majority class.
- Reply: One fixed canned response.
- Escalation: Simple keyword match.
"""
from __future__ import annotations

import re


class TrivialBaseline:
    def __init__(self, majority_intent: str = "delivery_issue"):
        self.majority_intent = majority_intent
        self.canned_reply = (
            "I'm sorry you are experiencing this issue. "
            "Please send us a DM with your order number so we can look into it."
        )
        self.escalation_keywords = [
            'lawsuit', 'refund now', 'unacceptable', 'speak to manager', 
            'attorney', 'legal action', 'chargeback'
        ]
        
    def predict(self, message: str, thread_history: list[dict]) -> dict:
        """Predict outcomes for a single message."""
        message_lower = message.lower()
        
        # Simple keyword matching for escalation
        escalate = False
        reason = ""
        for kw in self.escalation_keywords:
            if kw in message_lower:
                escalate = True
                reason = f"Keyword match: '{kw}'"
                break
                
        return {
            'intent': self.majority_intent,
            'draft_reply': self.canned_reply,
            'grounding_sources': ["N/A - Trivial Baseline"],
            'retrieval_confidence': 0.0,
            'escalate': escalate,
            'escalate_reason': reason,
            'ood_flag': False,
            'truncation_log': {}
        }
