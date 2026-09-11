"""Intent Classifier.

Uses k-NN logic backed by the SimpleBaseline's fitted models, but wrapped 
for the Agent Pipeline.
"""
from __future__ import annotations

class IntentClassifier:
    def __init__(self, llm=None, knn=None, knn_intents=None, embedder=None):
        self.llm = llm
        self.knn = knn
        self.knn_intents = knn_intents
        self.embedder = embedder
        
    def predict(self, message: str) -> str:
        if not self.embedder or not self.knn or not self.knn_intents:
            return "unknown"
            
        emb = self.embedder.encode([message])[0]
        distances, indices = self.knn.kneighbors([emb], n_neighbors=1)
        best_idx = indices[0][0]
        return self.knn_intents[best_idx]
