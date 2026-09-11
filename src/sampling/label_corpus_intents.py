"""One-time offline script to label the full corpus with predicted intents using k-NN.

This resolves Issue #3: RAG filtering by intent needs intent labels on the entire corpus,
but we only manually label the ~200 golden examples. This script uses the golden set
to train a k-NN classifier (reusing the FAISS index embeddings) and labels the full corpus.

Usage:
    python -m src.sampling.label_corpus_intents --corpus path/to/clean_conversations.jsonl --golden path/to/golden_calibration.jsonl
"""
from __future__ import annotations

import json
import os
from collections import Counter

import click
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def get_customer_initial_message(conversation: dict) -> str:
    """Extract the first customer message."""
    for turn in conversation['turns']:
        if turn['role'] == 'customer':
            return turn['text']
    return ""

@click.command()
@click.option('--corpus-path', default='data/processed/clean_conversations.jsonl')
@click.option('--golden-path', default='data/golden/golden_calibration.jsonl') # Or combined golden
@click.option('--output-path', default='data/processed/labeled_corpus.jsonl')
@click.option('--model-name', default='all-MiniLM-L6-v2')
@click.option('--k-neighbors', default=5)
def main(corpus_path: str, golden_path: str, output_path: str, model_name: str, k_neighbors: int):
    
    print(f"Loading golden set from {golden_path}...")
    if not os.path.exists(golden_path):
        print(f"ERROR: Golden set not found at {golden_path}.")
        print("Run this script AFTER manual labeling of the golden set.")
        return

    golden_convs = []
    with open(golden_path) as f:
        for line in f:
            golden_convs.append(json.loads(line))
            
    print(f"Loaded {len(golden_convs)} golden examples.")
    
    # Extract training data (X = embeddings of first message, y = intent)
    X_train_texts = []
    y_train = []
    
    for conv in golden_convs:
        # We need manually labeled intent
        if 'intent' not in conv or not conv['intent']:
            print("ERROR: Golden set must contain 'intent' labels.")
            return
            
        text = get_customer_initial_message(conv)
        X_train_texts.append(text)
        y_train.append(conv['intent'])
        
    print(f"Intent distribution in training data: {Counter(y_train)}")
    
    print(f"Loading embedding model {model_name}...")
    embedder = SentenceTransformer(model_name)
    
    print("Embedding training data...")
    X_train_emb = embedder.encode(X_train_texts, show_progress_bar=True)
    
    print(f"Training k-NN classifier (k={k_neighbors})...")
    knn = KNeighborsClassifier(n_neighbors=min(k_neighbors, len(set(y_train))))
    knn.fit(X_train_emb, y_train)
    
    # Log calibration accuracy (CV)
    cv_scores = cross_val_score(knn, X_train_emb, y_train, cv=3)
    print(f"Estimated calibration accuracy (3-fold CV): {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    print(f"\nLoading full corpus from {corpus_path}...")
    corpus = []
    with open(corpus_path) as f:
        for line in f:
            corpus.append(json.loads(line))
            
    print(f"Extracting queries from {len(corpus):,} conversations...")
    queries = [get_customer_initial_message(c) for c in corpus]
    
    # Process in batches to avoid OOM
    batch_size = 5000
    predicted_intents = []
    
    print("Embedding and predicting corpus intents...")
    for i in tqdm(range(0, len(queries), batch_size)):
        batch_queries = queries[i:i+batch_size]
        batch_emb = embedder.encode(batch_queries, show_progress_bar=False)
        batch_preds = knn.predict(batch_emb)
        predicted_intents.extend(batch_preds)
        
    # Attach predictions
    for c, intent in zip(corpus, predicted_intents):
        c['predicted_intent'] = intent
        
    print(f"\nPredicted intent distribution: {Counter(predicted_intents)}")
    
    print(f"\nSaving labeled corpus to {output_path}...")
    with open(output_path, 'w') as f:
        for c in corpus:
            f.write(json.dumps(c) + '\n')
            
    print("Done!")

if __name__ == '__main__':
    main()
