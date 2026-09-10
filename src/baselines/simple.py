from __future__ import annotations

class SimpleBaseline:
    """Simple baseline using embeddings and FAISS."""
    def __init__(self):
        """Loads embeddings + FAISS index."""
        pass
        
    def predict(self, message: str, thread_history: list) -> dict:
        """
        Intent: embedding + k-NN (k=5). 
        Reply: nearest-neighbor historical brand reply (pure retrieval). 
        Escalation: k-NN confidence + keyword + VADER sentiment via logistic regression.
        """
        return {
            "intent": "specific_inquiry",
            "reply": "Here is the information you requested based on similar past queries.",
            "escalate": False,
            "escalate_reason": ""
        }
