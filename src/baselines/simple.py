"""Simple Baseline.

Uses traditional ML and heuristics, no LLM generation:
- Intent: k-NN (k=5) using sentence-transformers.
- Reply: Pure retrieval (returns the actual historical support reply from the nearest neighbor).
- Escalation: Logistic regression over (k-NN distance, sentiment, length).
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SimpleBaseline:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.embedder = SentenceTransformer(model_name)
        self.analyzer = SentimentIntensityAnalyzer()
        self.knn = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.logreg = LogisticRegression(class_weight='balanced')
        
        self.is_fitted = False
        self.corpus_texts = []
        self.corpus_replies = []
        self.corpus_intents = []
        self.corpus_embeddings = None
        
    def fit_corpus(self, corpus_path: str, intent_labels_path: str | None = None):
        """Build the retrieval corpus (resolves Issue #1 leakage if held_out used)."""
        print(f"Loading corpus for Simple Baseline from {corpus_path}...")
        
        # Load held out IDs to prevent leakage (Issue #1 fix)
        held_out = set()
        if os.path.exists('data/golden/held_out_ids.json'):
            with open('data/golden/held_out_ids.json') as f:
                held_out = set(json.load(f))
                
        with open(corpus_path) as f:
            for line in f:
                c = json.loads(line)
                if c['conversation_id'] in held_out:
                    continue
                    
                # Extract first customer message and first support reply
                cust_msg = ""
                supp_reply = ""
                for t in c['turns']:
                    if t['role'] == 'customer' and not cust_msg:
                        cust_msg = t['text']
                    elif t['role'] == 'support' and not supp_reply:
                        supp_reply = t['text']
                        
                if cust_msg and supp_reply:
                    self.corpus_texts.append(cust_msg)
                    self.corpus_replies.append(supp_reply)
                    self.corpus_intents.append(c.get('predicted_intent', 'unknown'))
                    
        print(f"Embedding {len(self.corpus_texts)} corpus queries...")
        self.corpus_embeddings = self.embedder.encode(self.corpus_texts, show_progress_bar=True)
        self.knn.fit(self.corpus_embeddings)
        
    def fit_escalation(self, calibration_path: str):
        """Fit the escalation LogisticRegression on the calibration split only (Issue #7 fix)."""
        print(f"Fitting escalation logic on {calibration_path}...")
        
        X = []
        y = []
        
        with open(calibration_path) as f:
            for line in f:
                c = json.loads(line)
                cust_msg = ""
                for t in c['turns']:
                    if t['role'] == 'customer':
                        cust_msg = t['text']
                        break
                        
                # Extract features
                features = self._extract_features(cust_msg)
                X.append(features)
                
                # We need golden escalation labels for fitting!
                # If they don't exist yet (before manual labeling), we mock them
                label = c.get('escalate', False)
                y.append(int(label))
                
        # Handle case where all labels are identical (e.g. before manual labeling)
        if len(set(y)) < 2:
            print("WARNING: Calibration data lacks diverse escalation labels. Using dummy weights.")
            # Dummy fit just so it doesn't crash
            X = [[0, 0, 0], [1, 1, 1]]
            y = [0, 1]
            
        self.logreg.fit(X, y)
        self.is_fitted = True
        
    def _extract_features(self, message: str) -> list[float]:
        """Extract features for the logistic regression."""
        # 1. k-NN distance (proxy for OOD/weirdness)
        emb = self.embedder.encode([message])[0]
        distances, _ = self.knn.kneighbors([emb], n_neighbors=1)
        knn_dist = float(distances[0][0])
        
        # 2. VADER sentiment (negative sentiment = higher escalation risk)
        sentiment = self.analyzer.polarity_scores(message)['compound']
        
        # 3. Message length
        length = len(message.split())
        
        return [knn_dist, sentiment, length]
        
    def predict(self, message: str, thread_history: list[dict]) -> dict:
        if not self.is_fitted:
            raise RuntimeError("SimpleBaseline must be fitted before prediction.")
            
        # 1. Retrieval & Intent
        emb = self.embedder.encode([message])[0]
        distances, indices = self.knn.kneighbors([emb], n_neighbors=1)
        
        best_idx = indices[0][0]
        distance = distances[0][0]
        retrieval_confidence = max(0.0, 1.0 - distance) # roughly convert cosine dist to sim
        
        intent = self.corpus_intents[best_idx]
        reply = self.corpus_replies[best_idx]
        
        # 2. Escalation
        features = self._extract_features(message)
        escalate_prob = self.logreg.predict_proba([features])[0][1]
        escalate = bool(escalate_prob > 0.5)
        
        reason = f"LogReg prob {escalate_prob:.2f} > 0.5" if escalate else ""
        
        return {
            'intent': intent,
            'draft_reply': reply,
            'grounding_sources': [self.corpus_texts[best_idx]],
            'retrieval_confidence': retrieval_confidence,
            'escalate': escalate,
            'escalate_reason': reason,
            'ood_flag': distance > 0.5, # Simple threshold
            'truncation_log': {}
        }
