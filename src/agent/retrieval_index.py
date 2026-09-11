"""FAISS Retrieval Index.

Builds a dense retrieval index over the customer support corpus.
Crucially, it EXCLUDES any conversations present in `held_out_ids.json`
to prevent data leakage into the evaluation phase (Issue #1 fix).

Usage:
    python -m src.agent.retrieval_index --build
"""
from __future__ import annotations

import json
import os
import pickle
from typing import Optional

import click
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class RetrievalIndex:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []
        
    def build(self, corpus_path: str, held_out_path: str | None = 'data/golden/held_out_ids.json'):
        """Build FAISS index from corpus, explicitly excluding held-out IDs."""
        print(f"Loading corpus from {corpus_path}...")
        
        # Load held-out IDs
        held_out = set()
        if held_out_path and os.path.exists(held_out_path):
            with open(held_out_path) as f:
                held_out = set(json.load(f))
            print(f"Loaded {len(held_out)} held-out IDs to exclude (Issue #1 fix).")
            
        valid_conversations = []
        with open(corpus_path) as f:
            for line in f:
                c = json.loads(line)
                if c['conversation_id'] in held_out:
                    continue
                valid_conversations.append(c)
                
        print(f"Building index over {len(valid_conversations):,} non-held-out conversations...")
        
        # Extract queries (first customer message) and payloads
        texts_to_embed = []
        self.metadata = []
        
        for c in valid_conversations:
            cust_msg = ""
            for t in c['turns']:
                if t['role'] == 'customer':
                    cust_msg = t['text']
                    break
                    
            if not cust_msg:
                continue
                
            texts_to_embed.append(cust_msg)
            self.metadata.append({
                'conversation_id': c['conversation_id'],
                'intent': c.get('predicted_intent', 'unknown'),
                'customer_query': cust_msg,
                'full_thread': c['turns']
            })
            
        print("Embedding queries...")
        # Batch encode to avoid OOM
        embeddings = self.embedder.encode(texts_to_embed, show_progress_bar=True)
        
        print("Building FAISS index...")
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d) # Inner product (cosine sim since vectors are usually normalized)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        print(f"Index built with {self.index.ntotal} vectors.")
        
    def save(self, index_path: str, meta_path: str):
        """Save FAISS index and metadata to disk."""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"Saved index to {index_path} and metadata to {meta_path}")
        
    def load(self, index_path: str, meta_path: str):
        """Load FAISS index and metadata from disk."""
        self.index = faiss.read_index(index_path)
        with open(meta_path, 'rb') as f:
            self.metadata = pickle.load(f)
            
    def search(self, query: str, top_k: int = 5, intent_filter: Optional[str] = None) -> list[dict]:
        """
        Search for most similar historical threads.
        If intent_filter is provided, returns the top_k results matching that intent.
        """
        if self.index is None:
            raise RuntimeError("Index not loaded or built.")
            
        emb = self.embedder.encode([query])
        faiss.normalize_L2(emb)
        
        # Retrieve more than k if we need to filter
        search_k = top_k * 5 if intent_filter else top_k
        distances, indices = self.index.search(emb, search_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                break
                
            meta = self.metadata[idx]
            
            if intent_filter and meta['intent'] != intent_filter:
                continue
                
            results.append({
                'conversation_id': meta['conversation_id'],
                'intent': meta['intent'],
                'similarity': float(dist),
                'customer_query': meta['customer_query'],
                'full_thread': meta['full_thread']
            })
            
            if len(results) >= top_k:
                break
                
        return results


@click.command()
@click.option('--build', is_flag=True, help="Build the index from corpus")
@click.option('--corpus-path', default='data/processed/labeled_corpus.jsonl')
@click.option('--index-path', default='data/index/faiss.index')
@click.option('--meta-path', default='data/index/metadata.pkl')
def main(build: bool, corpus_path: str, index_path: str, meta_path: str):
    idx = RetrievalIndex()
    if build:
        idx.build(corpus_path)
        idx.save(index_path, meta_path)
    else:
        print("Loading existing index to test search...")
        idx.load(index_path, meta_path)
        res = idx.search("Where is my package?", top_k=2)
        print(json.dumps(res, indent=2))

if __name__ == '__main__':
    main()
