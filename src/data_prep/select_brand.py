"""Brand selection with mechanical deflection-trap scoring.

Computes per-brand:
  - Lexical diversity (unique trigram ratio over outbound replies)
  - Self-similarity (mean pairwise embedding cosine sim of sampled replies)
  - Reply length distribution (median + IQR of word counts)
  - Resolution proxy rate (% of threads ending with brand reply + alternating turns)
  - Composite score = (lexical_diversity * resolution_proxy_rate) / self_similarity

Usage:
    python -m src.data_prep.select_brand [--top-n 5] [--min-volume 5000]
"""
from __future__ import annotations

import os
import re
import random
from collections import Counter
from itertools import combinations

import click
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_conversation_turns(conversation_text: str) -> list[dict]:
    """Parse a conversation string into structured turns.
    
    Format: 'Customer: ... \nSupport: ... \nCustomer: ...'
    """
    turns = []
    if not isinstance(conversation_text, str) or not conversation_text.strip():
        return turns
    
    # Split on role prefixes
    pattern = r'(Customer|Support):\s*'
    parts = re.split(pattern, conversation_text)
    
    # parts[0] is empty or preamble, then alternating role/text
    i = 1
    while i < len(parts) - 1:
        role = parts[i].strip().lower()
        text = parts[i + 1].strip()
        if text:
            turns.append({'role': role, 'text': text})
        i += 2
    
    return turns


def get_support_replies(df: pd.DataFrame, brand: str) -> list[str]:
    """Extract all support (brand) replies for a given brand."""
    brand_df = df[df['company'] == brand]
    replies = []
    for _, row in brand_df.iterrows():
        turns = parse_conversation_turns(row['conversation'])
        for turn in turns:
            if turn['role'] == 'support':
                replies.append(turn['text'])
    return replies


def compute_trigram_diversity(texts: list[str]) -> float:
    """Compute unique trigram ratio across all texts.
    
    Low ratio = canned/templated responses (deflection trap).
    """
    all_trigrams = []
    for text in texts:
        words = text.lower().split()
        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        all_trigrams.extend(trigrams)
    
    if not all_trigrams:
        return 0.0
    
    return len(set(all_trigrams)) / len(all_trigrams)


def compute_self_similarity(texts: list[str], sample_size: int = 200, seed: int = 42) -> float:
    """Compute mean pairwise cosine similarity of a sample of replies.
    
    Uses TF-IDF vectors (fast, no model download needed for brand selection).
    High self-similarity = canned/templated responses.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    if len(texts) < 10:
        return 1.0  # Not enough data
    
    rng = random.Random(seed)
    sample = rng.sample(texts, min(sample_size, len(texts)))
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sample)
    
    # Compute pairwise similarities on a subset to avoid O(n^2) explosion
    n = min(100, len(sample))
    sim_matrix = cosine_similarity(tfidf_matrix[:n])
    
    # Extract upper triangle (exclude diagonal)
    upper_indices = np.triu_indices(n, k=1)
    similarities = sim_matrix[upper_indices]
    
    return float(np.mean(similarities))


def compute_reply_length_stats(texts: list[str]) -> dict:
    """Compute median and IQR of reply word counts."""
    lengths = [len(t.split()) for t in texts]
    if not lengths:
        return {'median': 0, 'iqr': 0, 'mean': 0}
    
    arr = np.array(lengths)
    q25, q75 = np.percentile(arr, [25, 75])
    return {
        'median': float(np.median(arr)),
        'iqr': float(q75 - q25),
        'mean': float(np.mean(arr)),
    }


def compute_resolution_proxy(df: pd.DataFrame, brand: str) -> float:
    """Compute resolution proxy rate for a brand.
    
    A conversation is 'resolved' if:
    - It has >= 2 alternating turns
    - The last message is from support (brand had the final word)
    - OR the conversation has >= 3 turns with alternation
    """
    brand_df = df[df['company'] == brand]
    total = 0
    resolved = 0
    
    for _, row in brand_df.iterrows():
        turns = parse_conversation_turns(row['conversation'])
        if len(turns) < 2:
            continue
        
        total += 1
        
        # Check for alternating turns (real conversation, not monologue)
        has_alternation = False
        for i in range(1, len(turns)):
            if turns[i]['role'] != turns[i-1]['role']:
                has_alternation = True
                break
        
        if not has_alternation:
            continue
        
        # Last turn is from support = brand had final word
        last_is_support = turns[-1]['role'] == 'support'
        
        # At least 2 alternating exchanges
        role_changes = sum(1 for i in range(1, len(turns)) if turns[i]['role'] != turns[i-1]['role'])
        
        if last_is_support and role_changes >= 2:
            resolved += 1
    
    return resolved / total if total > 0 else 0.0


def compute_brand_stats(
    data_path: str,
    min_volume: int = 5000,
    top_n: int = 10,
) -> pd.DataFrame:
    """Compute deflection-trap scores for all brands meeting volume threshold.
    
    Returns a DataFrame with per-brand metrics and composite score.
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Filter to brands with sufficient volume
    brand_counts = df['company'].value_counts()
    eligible_brands = brand_counts[brand_counts >= min_volume].index.tolist()
    
    # Remove the empty-string brand
    eligible_brands = [b for b in eligible_brands if b.strip()]
    
    print(f"Eligible brands (>= {min_volume} conversations): {len(eligible_brands)}")
    
    results = []
    
    for brand in tqdm(eligible_brands[:top_n], desc="Scoring brands"):
        print(f"\n--- {brand} ---")
        
        # Get support replies
        replies = get_support_replies(df, brand)
        print(f"  Support replies: {len(replies):,}")
        
        if len(replies) < 50:
            print(f"  SKIP: too few replies")
            continue
        
        # 1. Lexical diversity
        sample_for_diversity = random.sample(replies, min(2000, len(replies)))
        lex_div = compute_trigram_diversity(sample_for_diversity)
        print(f"  Lexical diversity: {lex_div:.4f}")
        
        # 2. Self-similarity
        self_sim = compute_self_similarity(replies)
        print(f"  Self-similarity: {self_sim:.4f}")
        
        # 3. Reply length stats
        length_stats = compute_reply_length_stats(replies)
        print(f"  Reply length median: {length_stats['median']:.1f}, IQR: {length_stats['iqr']:.1f}")
        
        # 4. Resolution proxy
        res_rate = compute_resolution_proxy(df, brand)
        print(f"  Resolution proxy rate: {res_rate:.4f}")
        
        results.append({
            'brand': brand,
            'conversation_count': int(brand_counts[brand]),
            'support_reply_count': len(replies),
            'lexical_diversity': round(lex_div, 4),
            'self_similarity': round(self_sim, 4),
            'reply_length_median': round(length_stats['median'], 1),
            'reply_length_iqr': round(length_stats['iqr'], 1),
            'resolution_proxy_rate': round(res_rate, 4),
        })

    # Convert to dataframe for vectorized normalization
    stats_df = pd.DataFrame(results)
    
    # 5. Composite score (Issue #14 fix: Z-score normalization)
    # Avoid division by zero by adding small epsilon to std
    for col in ['lexical_diversity', 'resolution_proxy_rate', 'self_similarity']:
        stats_df[f'{col}_z'] = (stats_df[col] - stats_df[col].mean()) / (stats_df[col].std() + 1e-9)
    
    # Composite = Z(lex_div) + Z(res_rate) - Z(self_sim)
    stats_df['composite_score'] = stats_df['lexical_diversity_z'] + stats_df['resolution_proxy_rate_z'] - stats_df['self_similarity_z']
    
    # Drop temp columns
    stats_df = stats_df.drop(columns=['lexical_diversity_z', 'resolution_proxy_rate_z', 'self_similarity_z'])
    
    return stats_df.sort_values('composite_score', ascending=False)


def rank_brands(stats: pd.DataFrame) -> pd.DataFrame:
    """Rank brands by composite score and return sorted DataFrame."""
    return stats.sort_values('composite_score', ascending=False).reset_index(drop=True)


@click.command()
@click.option('--data-path', default='data/raw/conversations.parquet', help='Path to raw conversations parquet')
@click.option('--top-n', default=10, help='Number of top brands to analyze')
@click.option('--min-volume', default=5000, help='Minimum conversation volume')
@click.option('--output', default='data/processed/brand_stats.parquet', help='Output path')
def main(data_path: str, top_n: int, min_volume: int, output: str):
    """Run brand selection with deflection-trap scoring."""
    stats = compute_brand_stats(data_path, min_volume=min_volume, top_n=top_n)
    ranked = rank_brands(stats)
    
    os.makedirs(os.path.dirname(output), exist_ok=True)
    ranked.to_parquet(output, index=False)
    
    print("\n" + "=" * 80)
    print("BRAND RANKING (by composite score)")
    print("=" * 80)
    print(ranked.to_string(index=False))
    print(f"\nSaved to {output}")
    
    # Recommendation
    if len(ranked) > 0:
        winner = ranked.iloc[0]
        print(f"\n✅ RECOMMENDED: {winner['brand']}")
        print(f"   Composite score: {winner['composite_score']}")
        print(f"   Conversations: {winner['conversation_count']:,}")
        print(f"   Lexical diversity: {winner['lexical_diversity']}")
        print(f"   Self-similarity: {winner['self_similarity']}")
        print(f"   Resolution rate: {winner['resolution_proxy_rate']}")


if __name__ == '__main__':
    main()
