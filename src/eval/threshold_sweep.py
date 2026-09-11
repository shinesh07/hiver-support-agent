"""Threshold Sweeping Logic.

Fits the confidence thresholds using ONLY the calibration split.
Includes critical statistical fixes:
- Issue #2: Selects the *smallest* threshold where Recall >= 0.90.
- Issue #5: Uses a shrinkage estimator for per-intent thresholds.

Usage:
    python -m src.eval.threshold_sweep
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
import numpy as np
import click


def _compute_metrics_at_t(similarities: list[float], groundings: list[bool], t: float) -> dict:
    """
    Computes Escalation metrics for a given threshold T.
    
    If similarity < T, we escalate (prediction = True).
    The critical positive class we MUST catch is ungrounded (grounding == False).
    """
    if not similarities:
        return {'recall': 0.0, 'precision': 0.0, 'f1': 0.0, 'escalation_rate': 0.0}
        
    n = len(similarities)
    
    # Ground truth: Should we escalate? (Yes if NOT grounded)
    y_true = [not g for g in groundings]
    
    # Prediction: Do we escalate? (Yes if similarity < T)
    y_pred = [sim < t for sim in similarities]
    
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt and not yp)
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    return {
        'recall': recall,
        'precision': precision,
        'escalation_rate': (tp + fp) / n
    }

def find_optimal_threshold(
    similarities: list[float], 
    groundings: list[bool], 
    target_recall: float = 0.90
) -> float:
    """
    Fix for Issue #2: We want the *smallest* T that achieves target_recall.
    Since escalating happens when similarity < T, higher T increases recall but hurts precision.
    We sort T candidates and find the first (smallest) one that satisfies the condition.
    """
    if not similarities:
        return 0.5
        
    # Test a dense grid of thresholds between 0 and 1
    t_candidates = np.linspace(0.0, 1.0, 101)
    
    best_t = 1.0 # Default to max escalation if we can't meet recall
    
    for t in t_candidates:
        metrics = _compute_metrics_at_t(similarities, groundings, t)
        if metrics['recall'] >= target_recall:
            best_t = t
            break # Since candidates are sorted ascending, this is the smallest valid T
            
    return float(best_t)

def fit_thresholds(calibration_path: str, target_recall: float = 0.90, margin_width: float = 0.05) -> dict:
    """
    Fits global and per-intent thresholds on calibration data.
    Implements Shrinkage Estimator for intents (Issue #5 fix).
    """
    print(f"Loading calibration set from {calibration_path}...")
    
    # In a real run, these similarities would come from running FAISS retrieval on the cal set.
    # We will simulate the retrieval step just for the calibration set here if they don't have it,
    # but for simplicity, we can load the golden set and run actual FAISS.
    
    # We need the FAISS index
    from src.agent.retrieval_index import RetrievalIndex
    retriever = RetrievalIndex()
    retriever.load('data/index/faiss.index', 'data/index/metadata.pkl')
    
    all_sims = []
    all_groundings = []
    
    intent_sims = defaultdict(list)
    intent_groundings = defaultdict(list)
    
    with open(calibration_path) as f:
        for line in f:
            c = json.loads(line)
            
            # Extract first customer message
            query = ""
            for turn in c['turns']:
                if turn['role'] == 'customer':
                    query = turn['text']
                    break
                    
            intent = c['intent']
            grounded = c['grounding_available']
            
            # Get similarity from FAISS
            res = retriever.search(query, top_k=1, intent_filter=intent)
            if not res:
                res = retriever.search(query, top_k=1)
                
            sim = res[0]['similarity'] if res else 0.0
            
            all_sims.append(sim)
            all_groundings.append(grounded)
            
            intent_sims[intent].append(sim)
            intent_groundings[intent].append(grounded)
            
    # Fit Global Threshold (Issue #2 applied here)
    global_t = find_optimal_threshold(all_sims, all_groundings, target_recall)
    
    # Fit Per-Intent Thresholds with Shrinkage (Issue #5 applied here)
    per_intent = {}
    
    for intent, sims in intent_sims.items():
        local_t = find_optimal_threshold(sims, intent_groundings[intent], target_recall)
        
        n_local = len(sims)
        # Shrinkage factor alpha: Approaches 1 as n increases.
        # If n=1, alpha=0.1. If n=10, alpha=0.5. If n=50, alpha=0.83
        n_prior = 10.0 
        alpha = n_local / (n_local + n_prior)
        
        shrunk_t = (alpha * local_t) + ((1 - alpha) * global_t)
        per_intent[intent] = shrunk_t
        
        print(f"Intent '{intent}': n={n_local:2d} | Local T: {local_t:.3f} | Global T: {global_t:.3f} -> Shrunk T: {shrunk_t:.3f}")
        
    config = {
        'global': global_t,
        'per_intent': per_intent,
        'margin_width': margin_width
    }
    
    print(f"\nFinal Global Threshold: {global_t:.3f}")
    
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/thresholds.json', 'w') as f:
        json.dump(config, f, indent=2)
        
    print(f"Saved threshold config to data/processed/thresholds.json")
    return config


@click.command()
@click.option('--calibration-path', default='data/golden/golden_calibration.jsonl')
@click.option('--target-recall', default=0.90)
@click.option('--margin-width', default=0.05)
def main(calibration_path: str, target_recall: float, margin_width: float):
    fit_thresholds(calibration_path, target_recall, margin_width)

if __name__ == '__main__':
    main()
