from __future__ import annotations

import click
import numpy as np
from sklearn.cluster import KMeans

def cluster_messages(conversations: list[dict], n_clusters: int = 10, sample_size: int = 2000, seed: int = 42) -> tuple[np.ndarray, KMeans]:
    """
    Embeds customer-initial messages using sentence-transformers and runs k-means.
    Returns (cluster_labels, kmeans_model).
    """
    # Mock implementation
    labels = np.zeros(len(conversations), dtype=int)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    return labels, kmeans

@click.command()
@click.option('--input-file', required=True, help="Path to clean conversations")
def cli(input_file: str):
    """CLI for clustering messages."""
    pass

if __name__ == '__main__':
    cli()
