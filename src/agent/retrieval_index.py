from __future__ import annotations

class RetrievalIndex:
    """FAISS index management."""
    def __init__(self):
        pass
        
    def build(self, conversations: list[dict]):
        """Builds index from conversations using sentence-transformers and FAISS."""
        pass
        
    def save(self, path: str):
        """Saves index to disk."""
        pass
        
    def load(self, path: str):
        """Loads index from disk."""
        pass
        
    def search(self, query: str, intent_filter: str | None, top_k: int = 5) -> list[dict]:
        """
        Searches index for similar conversations.
        Returns [{text, similarity, intent, resolution_flag}]
        """
        return []
