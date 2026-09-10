from __future__ import annotations

import numpy as np

class OODDetector:
    """OOD detection."""
    def __init__(self):
        """Loads cluster centroids."""
        self.centroids = None
        self.threshold = 0.5 # Placeholder for 95th percentile of within-cluster distances
        
    def is_ood(self, query_embedding: np.ndarray) -> tuple[bool, float]:
        """
        Returns (is_ood, distance_to_nearest_centroid)
        Threshold: 95th percentile of within-cluster distances
        """
        # Mock implementation
        distance = 0.4
        return distance > self.threshold, distance
