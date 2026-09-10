from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def compute_intent_metrics(predictions: list[str], labels: list[str]) -> dict:
    """Computes automated metrics for intent classification."""
    labels_unique = list(set(labels) | set(predictions))
    acc = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    per_class_f1_array = f1_score(labels, predictions, average=None, labels=labels_unique, zero_division=0)
    per_class_f1 = dict(zip(labels_unique, per_class_f1_array))
    cm = confusion_matrix(labels, predictions, labels=labels_unique).tolist()
    
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm
    }

def compute_escalation_metrics(predictions: list[bool], labels: list[bool]) -> dict:
    """Computes automated metrics for escalation logic."""
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    # False auto handle rate: predicted False, label True
    # False escalate rate: predicted True, label False
    fn = sum(1 for p, l in zip(predictions, labels) if not p and l)
    fp = sum(1 for p, l in zip(predictions, labels) if p and not l)
    tn = sum(1 for p, l in zip(predictions, labels) if not p and not l)
    tp = sum(1 for p, l in zip(predictions, labels) if p and l)
    
    false_auto_handle_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    false_escalate_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_auto_handle_rate": false_auto_handle_rate,
        "false_escalate_rate": false_escalate_rate
    }

def compute_latency_cost(results: list[dict]) -> dict:
    """Computes latency and cost metrics from a list of results."""
    latencies = [r.get("latency_ms", 0) for r in results]
    costs = [r.get("cost_usd", 0.0) for r in results]
    
    return {
        "latency_p50": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "latency_p95": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "mean_cost_per_request": float(np.mean(costs)) if costs else 0.0,
        "total_cost": sum(costs)
    }
