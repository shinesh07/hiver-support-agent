"""Build stratified golden set with calibration/evaluation split.

Samples from clustered data, ensuring representation across:
  - Intent clusters (proxy)
  - Thread length (short/med/long)
  - Difficulty (oversamples hard cases)

Splits into 30% calibration (threshold fitting) and 70% evaluation (metrics).
Also saves `held_out_ids.json` to prevent train-test leakage into the retrieval index (Issue #1).

Usage:
    python -m src.sampling.build_golden_set
"""
from __future__ import annotations

import json
import os
import random
import re

import click
import pandas as pd


def determine_difficulty(conversation: dict) -> str:
    """Classify conversation difficulty using heuristic signals."""
    all_text = ' '.join(t['text'].lower() for t in conversation['turns'])
    customer_text = ' '.join(t['text'].lower() for t in conversation['turns'] if t['role'] == 'customer')
    
    # Very short, ambiguous customer initial messages
    first_customer = ""
    for t in conversation['turns']:
        if t['role'] == 'customer':
            first_customer = t['text']
            break
            
    if len(first_customer.split()) <= 4:
        return 'hard'
        
    # Escalation keywords
    hard_keywords = [
        'refund', 'chargeback', 'lawsuit', 'lawyer', 'attorney', 'manager',
        'supervisor', 'unacceptable', 'furious', 'worst', 'stolen', 'hacked',
        'locked out', 'fraud'
    ]
    if any(k in customer_text for k in hard_keywords):
        return 'hard'
        
    # Profanity/anger markers
    if re.search(r'\b(wtf|bs|bullshit|scam|terrible|horrible|useless)\b', customer_text):
        return 'hard'
        
    # Excessive punctuation or ALL CAPS
    if '!!' in customer_text or '??' in customer_text:
        return 'hard'
    
    alpha_chars = sum(1 for c in customer_text if c.isalpha())
    upper_chars = sum(1 for c in customer_text if c.isupper())
    if alpha_chars > 20 and upper_chars / alpha_chars > 0.4:
        return 'hard'
        
    # High turn count (dragged-out issue)
    if conversation['turn_count'] >= 6:
        return 'hard'
        
    return 'easy'


def get_strata(conversation: dict) -> str:
    """Create a composite strata key for sampling."""
    cluster = conversation.get('cluster', 0)
    tc = conversation['turn_count']
    length_bucket = '1' if tc == 1 else '2-3' if tc <= 3 else '4-6' if tc <= 6 else '7+'
    return f"{cluster}_{length_bucket}"


@click.command()
@click.option('--input-path', default='data/processed/clustered_sample.jsonl')
@click.option('--n-total', default=200)
@click.option('--hard-ratio', default=0.3)
@click.option('--cal-ratio', default=0.3)
@click.option('--seed', default=42)
def main(input_path: str, n_total: int, hard_ratio: float, cal_ratio: float, seed: int):
    random.seed(seed)
    
    print(f"Loading clustered sample from {input_path}...")
    conversations = []
    with open(input_path) as f:
        for line in f:
            c = json.loads(line)
            c['difficulty_tier'] = determine_difficulty(c)
            c['strata'] = get_strata(c)
            conversations.append(c)
            
    print(f"Total available: {len(conversations)}")
    
    # Split by difficulty
    hard_pool = [c for c in conversations if c['difficulty_tier'] == 'hard']
    easy_pool = [c for c in conversations if c['difficulty_tier'] == 'easy']
    
    print(f"Pool: {len(easy_pool)} easy, {len(hard_pool)} hard")
    
    n_hard = int(n_total * hard_ratio)
    n_easy = n_total - n_hard
    
    # Sample from each pool (stratified by cluster_length bucket if possible)
    # Since n=200 is small, simple random sample within difficulty tiers is often sufficient,
    # but let's try to ensure diversity
    
    def stratified_draw(pool: list[dict], n_draw: int) -> list[dict]:
        df = pd.DataFrame([{'id': c['conversation_id'], 'strata': c['strata']} for c in pool])
        if len(df) == 0 or n_draw == 0:
            return []
            
        if n_draw >= len(df):
            return pool
            
        # Get counts per strata to compute weights
        strata_counts = df['strata'].value_counts()
        weights = df['strata'].map(lambda x: 1.0 / strata_counts[x])
        
        sampled_ids = df.sample(n=n_draw, weights=weights, random_state=seed)['id'].tolist()
        return [c for c in pool if c['conversation_id'] in sampled_ids]

    print(f"Drawing {n_hard} hard examples and {n_easy} easy examples...")
    sampled_hard = stratified_draw(hard_pool, n_hard)
    sampled_easy = stratified_draw(easy_pool, n_easy)
    
    golden_set = sampled_hard + sampled_easy
    random.shuffle(golden_set)
    
    # Split into Calibration and Evaluation
    # We must maintain the easy/hard ratio in both splits
    
    cal_hard_n = int(len(sampled_hard) * cal_ratio)
    cal_easy_n = int(len(sampled_easy) * cal_ratio)
    
    cal_hard = sampled_hard[:cal_hard_n]
    eval_hard = sampled_hard[cal_hard_n:]
    
    cal_easy = sampled_easy[:cal_easy_n]
    eval_easy = sampled_easy[cal_easy_n:]
    
    calibration = cal_hard + cal_easy
    evaluation = eval_hard + eval_easy
    
    random.shuffle(calibration)
    random.shuffle(evaluation)
    
    print(f"\nGolden set total: {len(golden_set)}")
    print(f"Calibration split: {len(calibration)} ({len(cal_hard)} hard, {len(cal_easy)} easy)")
    print(f"Evaluation split:  {len(evaluation)} ({len(eval_hard)} hard, {len(eval_easy)} easy)")
    
    # Save datasets
    os.makedirs('data/golden', exist_ok=True)
    
    with open('data/golden/golden_calibration.jsonl', 'w') as f:
        for c in calibration:
            f.write(json.dumps(c) + '\n')
            
    with open('data/golden/golden_evaluation.jsonl', 'w') as f:
        for c in evaluation:
            f.write(json.dumps(c) + '\n')
            
    # Fix for Issue #1: Save held-out IDs for retrieval index exclusion
    held_out_ids = [c['conversation_id'] for c in golden_set]
    with open('data/golden/held_out_ids.json', 'w') as f:
        json.dump(held_out_ids, f)
        
    print(f"\nSaved golden datasets to data/golden/")
    print(f"Saved {len(held_out_ids)} held_out_ids.json (Issue #1 fix)")

if __name__ == '__main__':
    main()
