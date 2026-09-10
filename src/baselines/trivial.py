from __future__ import annotations

class TrivialBaseline:
    """Trivial baseline for support agent."""
    def __init__(self):
        pass
        
    def predict(self, message: str, thread_history: list) -> dict:
        """
        Intent: majority class. 
        Reply: canned reply per intent. 
        Escalation: keyword list match.
        """
        return {
            "intent": "general_inquiry",
            "reply": "Thank you for reaching out. We will get back to you soon.",
            "escalate": False,
            "escalate_reason": ""
        }
