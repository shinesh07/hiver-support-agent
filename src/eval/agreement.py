"""Agreement and statistical significance testing.

Includes McNemar's test implementation for rigorous statistical comparison
between models (Issue #4 fix).
"""
from __future__ import annotations

import math
from typing import Sequence
import pandas as pd
from scipy.stats import chi2


def compute_mcnemars_test(y_true: Sequence[bool], y_model_a: Sequence[bool], y_model_b: Sequence[bool]) -> dict:
    """
    Computes McNemar's test to compare two classifiers on paired data.
    Resolves Issue #4 (bare 2% cutoff is statistically invalid).
    
    y_true: Ground truth binary labels
    y_model_a: Predictions from Model A (e.g., Simple Baseline)
    y_model_b: Predictions from Model B (e.g., Agent Pipeline)
    """
    if len(y_true) != len(y_model_a) or len(y_true) != len(y_model_b):
        raise ValueError("All arrays must be the same length")
        
    n = len(y_true)
    if n == 0:
        return {'statistic': 0.0, 'p_value': 1.0, 'significant': False, 'n': 0}
        
    # Get correctness vectors
    a_correct = [y_true[i] == y_model_a[i] for i in range(n)]
    b_correct = [y_true[i] == y_model_b[i] for i in range(n)]
    
    # Contingency table cells
    # n_01: B correct, A incorrect
    # n_10: A correct, B incorrect
    n_11 = sum(1 for i in range(n) if a_correct[i] and b_correct[i])
    n_10 = sum(1 for i in range(n) if a_correct[i] and not b_correct[i])
    n_01 = sum(1 for i in range(n) if not a_correct[i] and b_correct[i])
    n_00 = sum(1 for i in range(n) if not a_correct[i] and not b_correct[i])
    
    # McNemar's test statistic (with continuity correction)
    b = n_10
    c = n_01
    
    if b + c == 0:
        stat = 0.0
        p_val = 1.0
    else:
        stat = ((abs(b - c) - 1) ** 2) / (b + c)
        p_val = chi2.sf(stat, df=1)
        
    # Accuracy difference with 95% Confidence Interval (Wald interval)
    acc_diff = (c - b) / n
    se = math.sqrt((b + c) / (n ** 2) - ((c - b) ** 2) / (n ** 3))
    z = 1.96 # 95% CI
    ci_lower = acc_diff - z * se
    ci_upper = acc_diff + z * se
    
    return {
        'n': n,
        'accuracy_a': sum(a_correct) / n,
        'accuracy_b': sum(b_correct) / n,
        'accuracy_diff': acc_diff,
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper,
        'contingency_table': {'n_11': n_11, 'n_10': n_10, 'n_01': n_01, 'n_00': n_00},
        'statistic': stat,
        'p_value': p_val,
        'significant': p_val < 0.05
    }
