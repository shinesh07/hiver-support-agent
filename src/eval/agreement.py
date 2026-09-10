from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score

def compute_agreement(human_scores: list[dict], judge_scores: list[dict]) -> dict:
    """Computes human-judge agreement per dimension."""
    dimensions = ["faithfulness", "helpfulness", "tone", "conciseness", "safety"]
    agreement = {}
    
    for dim in dimensions:
        human_dim_scores = [hs.get(dim, 3) for hs in human_scores]
        judge_dim_scores = [js.get(dim, 3) for js in judge_scores]
        
        qwk = cohen_kappa_score(human_dim_scores, judge_dim_scores, weights='quadratic')
        exact_match = sum(1 for h, j in zip(human_dim_scores, judge_dim_scores) if h == j) / max(len(human_dim_scores), 1)
        within_one = sum(1 for h, j in zip(human_dim_scores, judge_dim_scores) if abs(h - j) <= 1) / max(len(human_dim_scores), 1)
        mean_bias = np.mean(np.array(judge_dim_scores) - np.array(human_dim_scores)) if human_dim_scores else 0.0
        
        agreement[dim] = {
            "quadratic_weighted_kappa": float(qwk),
            "exact_match_pct": exact_match,
            "within_one_pct": within_one,
            "mean_bias": float(mean_bias)
        }
        
    return agreement

def stratified_agreement(human_scores: list[dict], judge_scores: list[dict], difficulty_tiers: list[str]) -> dict:
    """Breaks down agreement by easy/hard tier."""
    result = {}
    unique_tiers = set(difficulty_tiers)
    
    for tier in unique_tiers:
        tier_indices = [i for i, t in enumerate(difficulty_tiers) if t == tier]
        tier_human = [human_scores[i] for i in tier_indices]
        tier_judge = [judge_scores[i] for i in tier_indices]
        result[tier] = compute_agreement(tier_human, tier_judge)
        
    return result

def check_prompt_order_bias(original_scores: list[dict], reordered_scores: list[dict]) -> dict:
    """Compares scores with reordered rubric dimensions."""
    return compute_agreement(original_scores, reordered_scores)
