"""Thread reconstruction from pre-parsed conversation data.

Since we're using the TNE-AI dataset which already has conversations
reconstructed, this module parses the conversation text into structured
turns and applies quality filters.

Usage:
    python -m src.data_prep.reconstruct_threads --brand AmazonHelp
"""
from __future__ import annotations

import re
import hashlib
from typing import Optional

import click
import pandas as pd
from tqdm import tqdm


def parse_turns(conversation_text: str) -> list[dict]:
    """Parse a conversation string into structured turns.
    
    Input format: 'Customer: text\\nSupport: text\\nCustomer: text'
    Some turns have agent signatures like ^MH, ^RS at the end.
    
    Returns list of {role: 'customer'|'support', text: str}
    """
    turns = []
    if not isinstance(conversation_text, str) or not conversation_text.strip():
        return turns
    
    # Split on role prefixes — handles multiline content
    pattern = r'\n(?=(?:Customer|Support):)'
    segments = re.split(pattern, conversation_text.strip())
    
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        
        if segment.startswith('Customer:'):
            role = 'customer'
            text = segment[len('Customer:'):].strip()
        elif segment.startswith('Support:'):
            role = 'support'
            text = segment[len('Support:'):].strip()
        else:
            # Continuation of previous turn or malformed
            if turns:
                turns[-1]['text'] += ' ' + segment
            continue
        
        # Remove agent signature tags (e.g., ^MH, ^RS, ^ARC)
        text = re.sub(r'\s*\^[A-Z]{1,4}\s*$', '', text).strip()
        
        if text:
            turns.append({'role': role, 'text': text})
    
    return turns


def is_valid_conversation(turns: list[dict]) -> bool:
    """Check if a conversation is valid for our purposes.
    
    Requirements:
    - At least 2 turns
    - Has both customer and support turns
    - Not a monologue (has at least one role alternation)
    """
    if len(turns) < 2:
        return False
    
    roles = {t['role'] for t in turns}
    if 'customer' not in roles or 'support' not in roles:
        return False
    
    # Check for at least one alternation
    for i in range(1, len(turns)):
        if turns[i]['role'] != turns[i-1]['role']:
            return True
    
    return False


def compute_resolution_flag(turns: list[dict]) -> bool:
    """Determine if a conversation appears resolved.
    
    Resolved if:
    - Last turn is from support (brand had final word)
    - At least 2 role alternations (real back-and-forth)
    """
    if not turns:
        return False
    
    last_is_support = turns[-1]['role'] == 'support'
    alternations = sum(1 for i in range(1, len(turns)) 
                       if turns[i]['role'] != turns[i-1]['role'])
    
    return last_is_support and alternations >= 2


def build_conversation_records(
    df: pd.DataFrame,
    brand: str,
) -> list[dict]:
    """Convert raw dataframe rows into structured conversation records.
    
    Returns list of dicts ready for JSONL output:
    {conversation_id, brand, turns, turn_count, resolution_flag}
    """
    brand_df = df[df['company'] == brand].copy()
    print(f"Processing {len(brand_df):,} conversations for {brand}...")
    
    records = []
    skipped_invalid = 0
    
    for _, row in tqdm(brand_df.iterrows(), total=len(brand_df), desc="Parsing"):
        turns = parse_turns(row['conversation'])
        
        if not is_valid_conversation(turns):
            skipped_invalid += 1
            continue
        
        records.append({
            'conversation_id': row['conversation_id'],
            'brand': brand,
            'turns': turns,
            'turn_count': len(turns),
            'resolution_flag': compute_resolution_flag(turns),
        })
    
    print(f"  Valid conversations: {len(records):,}")
    print(f"  Skipped (invalid): {skipped_invalid:,}")
    print(f"  Resolution rate: {sum(r['resolution_flag'] for r in records) / len(records) * 100:.1f}%")
    
    return records


@click.command()
@click.option('--data-path', default='data/raw/conversations.parquet')
@click.option('--brand', default='AmazonHelp')
@click.option('--output', default='data/processed/reconstructed_threads.jsonl')
def main(data_path: str, brand: str, output: str):
    """Reconstruct and validate conversation threads for a brand."""
    import json, os
    
    df = pd.read_parquet(data_path)
    records = build_conversation_records(df, brand)
    
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    
    print(f"\nSaved {len(records):,} conversations to {output}")


if __name__ == '__main__':
    main()
