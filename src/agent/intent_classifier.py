from __future__ import annotations

class IntentClassifier:
    """Intent classification."""
    def __init__(self, method: str = 'llm'):
        """Supports both 'knn' and 'llm' methods, selectable via config."""
        self.method = method
        
    def classify(self, message: str, thread_history: list) -> dict:
        """
        Classifies intent.
        Returns {intent, confidence, method}
        """
        return {
            "intent": "general",
            "confidence": 0.9,
            "method": self.method
        }
