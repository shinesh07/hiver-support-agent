"""Cluster customer-initial messages for stratified sampling.

Usage:
    python -m src.sampling.cluster_for_stratification
"""
from __future__ import annotations

import json
import os
import random

import click
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

def get_customer_initial_message(conversation: dict) -> str:
    """Extract the first customer message."""
    for turn in conversation['turns']:
        if turn['role'] == 'customer':
            return turn['text']
    return ""

@click.command()
@click.option('--input-path', default='data/processed/clean_conversations.jsonl')
@click.option('--output-path', default='data/processed/clustered_sample.jsonl')
@click.option('--n-clusters', default=10)
@click.option('--sample-size', default=2000)
@click.option('--seed', default=42)
@click.option('--model-name', default='all-MiniLM-L6-v2')
def main(input_path: str, output_path: str, n_clusters: int, sample_size: int, seed: int, model_name: str):
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"Loading conversations from {input_path}...")
    conversations = []
    with open(input_path) as f:
        for line in f:
            conversations.append(json.loads(line))
            
    if len(conversations) > sample_size:
        print(f"Sampling {sample_size} conversations for clustering...")
        sample = random.sample(conversations, sample_size)
    else:
        sample = conversations
        
    print("Extracting customer initial messages...")
    queries = [get_customer_initial_message(c) for c in sample]
    
    print(f"Loading embedding model {model_name}...")
    embedder = SentenceTransformer(model_name)
    
    print("Embedding messages...")
    embeddings = embedder.encode(queries, show_progress_bar=True)
    
    print(f"Clustering with K-Means (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    print("Saving clustered sample...")
    for c, label, emb in zip(sample, cluster_labels, embeddings):
        c['cluster'] = int(label)
        c['distance_to_centroid'] = float(np.linalg.norm(emb - kmeans.cluster_centers_[label]))
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for c in sample:
            f.write(json.dumps(c) + '\n')
            
    # Compute 95th percentile distance for OOD detection later
    distances = [c['distance_to_centroid'] for c in sample]
    p95_distance = np.percentile(distances, 95)
    
    # Save centroids and p95 threshold for OOD detector
    np.save('data/processed/cluster_centroids.npy', kmeans.cluster_centers_)
    with open('data/processed/ood_threshold.json', 'w') as f:
        json.dump({'p95_distance': float(p95_distance)}, f)
        
    print(f"Cluster counts: {pd.Series(cluster_labels).value_counts().to_dict()}")
    print(f"OOD threshold (95th percentile distance): {p95_distance:.4f}")
    print(f"Centroids saved to data/processed/cluster_centroids.npy")
    print(f"Clustered sample saved to {output_path}")

if __name__ == '__main__':
    main()
