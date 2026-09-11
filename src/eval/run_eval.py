"""Run the full evaluation harness.

Evaluates the Trivial Baseline, Simple Baseline, and Agent Pipeline on the evaluation set.
Applies McNemar's test for statistical significance (Issue #4).

Usage:
    python -m src.eval.run_eval
"""
from __future__ import annotations

import json
import os
import time
from tqdm import tqdm

from src.baselines.trivial import TrivialBaseline
from src.baselines.simple import SimpleBaseline
from src.agent.retrieval_index import RetrievalIndex
from src.agent.intent_classifier import IntentClassifier
from src.agent.ood_detector import OODDetector
from src.agent.pipeline import classify_and_respond
from src.llm.provider_adapter import LLMProvider
from src.eval.automated_metrics import compute_intent_metrics, compute_escalation_metrics
from src.eval.agreement import compute_mcnemars_test

def load_dataset(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main():
    print("=" * 80)
    print("HIVER SUPPORT AGENT — EVALUATION HARNESS")
    print("=" * 80)
    
    # 1. Load Data
    eval_data = load_dataset('data/golden/golden_evaluation.jsonl')
    print(f"Loaded {len(eval_data)} evaluation examples.")
    
    # 2. Setup Baselines & Agent
    print("\nInitializing models...")
    
    trivial = TrivialBaseline()
    
    simple = SimpleBaseline()
    simple.fit_corpus('data/processed/labeled_corpus.jsonl')
    simple.fit_escalation('data/golden/golden_calibration.jsonl')
    
    llm = LLMProvider(mock_mode=True) # Use mock mode for speed during testing
    
    retriever = RetrievalIndex(embedder=simple.embedder)
    retriever.load('data/index/faiss.index', 'data/index/metadata.pkl')
    
    intent_classifier = IntentClassifier(llm=llm, knn=simple.knn, knn_intents=simple.corpus_intents, embedder=simple.embedder)
    
    ood_detector = OODDetector(embedder=simple.embedder)
    ood_detector.load('data/processed/cluster_centroids.npy', 'data/processed/ood_threshold.json')
    
    with open('data/processed/thresholds.json') as f:
        threshold_config = json.load(f)
        
    # 3. Ground Truth Vectors
    y_true_intent = [c['intent'] for c in eval_data]
    y_true_escalate = [c['escalate'] for c in eval_data]
    grounding_available = [c['grounding_available'] for c in eval_data]
    
    # 4. Predictions Vectors
    t_intent, t_escalate = [], []
    s_intent, s_escalate = [], []
    a_intent, a_escalate = [], []
    
    print("\nRunning evaluation sweep...")
    start_time = time.time()
    
    for c in tqdm(eval_data):
        # Extract first customer message
        query = ""
        for turn in c['turns']:
            if turn['role'] == 'customer':
                query = turn['text']
                break
                
        history = c['turns']
        
        # Trivial
        res_t = trivial.predict(query, history)
        t_intent.append(res_t['intent'])
        t_escalate.append(res_t['escalate'])
        
        # Simple
        res_s = simple.predict(query, history)
        s_intent.append(res_s['intent'])
        s_escalate.append(res_s['escalate'])
        
        # Agent
        res_a = classify_and_respond(
            message=query,
            thread_history=history,
            intent_classifier=intent_classifier,
            ood_detector=ood_detector,
            retriever=retriever,
            llm=llm,
            threshold_config=threshold_config
        )
        a_intent.append(res_a['intent'])
        a_escalate.append(res_a['escalate'])
        
    duration = time.time() - start_time
    print(f"Evaluation finished in {duration:.1f}s.")
    
    # 5. Compute Metrics
    print("\n" + "=" * 80)
    print("RESULTS: INTENT CLASSIFICATION")
    print("=" * 80)
    
    print(f"Trivial Baseline Accuracy: {compute_intent_metrics(y_true_intent, t_intent)['accuracy']:.3f}")
    print(f"Simple Baseline Accuracy:  {compute_intent_metrics(y_true_intent, s_intent)['accuracy']:.3f}")
    print(f"Agent Pipeline Accuracy:   {compute_intent_metrics(y_true_intent, a_intent)['accuracy']:.3f}")
    
    print("\nStatistical Significance (Simple vs Agent) - McNemar's Test:")
    mcnemar_intent = compute_mcnemars_test(y_true_intent, s_intent, a_intent)
    print(f"  Accuracy Diff: {mcnemar_intent['accuracy_diff']*100:+.1f}%")
    print(f"  95% CI:        [{mcnemar_intent['ci_95_lower']*100:.1f}%, {mcnemar_intent['ci_95_upper']*100:.1f}%]")
    print(f"  p-value:       {mcnemar_intent['p_value']:.4f} (Significant? {mcnemar_intent['significant']})")

    print("\n" + "=" * 80)
    print("RESULTS: ESCALATION & SAFETY")
    print("=" * 80)
    
    met_t = compute_escalation_metrics(y_true_escalate, t_escalate, grounding_available)
    met_s = compute_escalation_metrics(y_true_escalate, s_escalate, grounding_available)
    met_a = compute_escalation_metrics(y_true_escalate, a_escalate, grounding_available)
    
    print("Trivial Baseline:")
    print(f"  Recall (Ungrounded): {met_t['recall_ungrounded']:.3f} | Precision: {met_t['overall_precision']:.3f} | Esc. Rate: {met_t['escalation_rate']:.3f}")
    
    print("\nSimple Baseline:")
    print(f"  Recall (Ungrounded): {met_s['recall_ungrounded']:.3f} | Precision: {met_s['overall_precision']:.3f} | Esc. Rate: {met_s['escalation_rate']:.3f}")
    
    print("\nAgent Pipeline:")
    print(f"  Recall (Ungrounded): {met_a['recall_ungrounded']:.3f} | Precision: {met_a['overall_precision']:.3f} | Esc. Rate: {met_a['escalation_rate']:.3f}")
    
    print("\nStatistical Significance (Simple vs Agent Escalation Accuracy) - McNemar's Test:")
    mcnemar_esc = compute_mcnemars_test(y_true_escalate, s_escalate, a_escalate)
    print(f"  Accuracy Diff: {mcnemar_esc['accuracy_diff']*100:+.1f}%")
    print(f"  95% CI:        [{mcnemar_esc['ci_95_lower']*100:.1f}%, {mcnemar_esc['ci_95_upper']*100:.1f}%]")
    print(f"  p-value:       {mcnemar_esc['p_value']:.4f} (Significant? {mcnemar_esc['significant']})")
    
if __name__ == '__main__':
    main()
