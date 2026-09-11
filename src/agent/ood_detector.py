"""OOD Detector.

Checks if a message is out-of-distribution using cluster distances.
"""
from __future__ import annotations

import json
import numpy as np
from sentence_transformers import SentenceTransformer


class OODDetector:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', embedder=None):
        if embedder:
            self.embedder = embedder
        else:
            self.embedder = SentenceTransformer(model_name)
        self.centroids = None
        self.threshold = 1.0
        
    def load(self, centroids_path: str, threshold_path: str):
        self.centroids = np.load(centroids_path)
        with open(threshold_path) as f:
            self.threshold = json.load(f)['p95_distance']
            
    def check_ood(self, message: str) -> tuple[bool, float]:
        if self.centroids is None:
            return False, 0.0
            
        emb = self.embedder.encode([message])[0]
        # Find distance to nearest centroid
        distances = np.linalg.norm(self.centroids - emb, axis=1)
        min_dist = float(np.min(distances))
        
        # OOD if distance > threshold
        return min_dist > self.threshold, min_dist
