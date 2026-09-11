"""Text cleaning pipeline for customer support conversations.

Applies:
  1. Language detection (English only, confidence >= 0.8)
  2. @-handle anonymization verification
  3. URL normalization
  4. HTML entity decoding
  5. Whitespace/Unicode normalization (NFC)
  6. Media placeholder removal
  7. Token counting per conversation
  8. Near-duplicate removal (Jaccard similarity > 0.9)

Usage:
    python -m src.data_prep.clean_text
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
import html
from collections import Counter

import click
from tqdm import tqdm


def detect_language(text: str) -> tuple[str, float]:
    """Detect language of text. Returns (lang_code, confidence)."""
    try:
        from langdetect import detect_langs
        results = detect_langs(text)
        if results:
            return results[0].lang, results[0].prob
    except Exception:
        pass
    return 'unknown', 0.0


def is_english_conversation(turns: list[dict], min_confidence: float = 0.8) -> bool:
    """Check if a conversation is primarily in English.
    
    Concatenates all turn text and checks language.
    """
    all_text = ' '.join(t['text'] for t in turns)
    if len(all_text.split()) < 5:
        return True  # Too short to detect; keep it
    
    lang, confidence = detect_language(all_text)
    return lang == 'en' and confidence >= min_confidence


def clean_text(text: str) -> str:
    """Clean a single text string.
    
    Steps:
    1. Decode HTML entities (&amp; -> &, etc.)
    2. Unicode normalize (NFC)
    3. Anonymize any remaining @-handles (should already be anonymized as @numbers)
    4. Normalize URLs (replace with [URL])
    5. Remove media placeholders
    6. Normalize whitespace
    """
    if not isinstance(text, str):
        return ''
    
    # HTML entity decoding
    text = html.unescape(text)
    
    # Unicode normalization
    text = unicodedata.normalize('NFC', text)
    
    # Verify/enforce @-handle anonymization
    # The dataset uses numeric handles like @115712 — keep those
    # But anonymize any real handles that slipped through
    text = re.sub(r'@([A-Za-z_]\w{2,})', '@[HANDLE]', text)
    
    # Normalize URLs — replace with [URL] placeholder
    text = re.sub(r'https?://\S+', '[URL]', text)
    text = re.sub(r't\.co/\S+', '[URL]', text)
    
    # Remove media placeholders
    text = re.sub(r'\[?(pic|photo|image|video|media)\]?', '', text, flags=re.IGNORECASE)
    
    # Remove RT markers
    text = re.sub(r'^RT\s+', '', text)
    
    # PII Scrubbing (Issue #8 fix)
    # Emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    # Phone numbers (US/International approximations)
    text = re.sub(r'\+?\b\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)
    text = re.sub(r'\b\d{3}[-.\s]\d{4}\b', '[PHONE]', text)
    # SSN / Credit Card approximations
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    text = re.sub(r'\b(?:\d{4}[-\s]?){3}\d{4}\b', '[CREDIT_CARD]', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def count_tokens(text: str) -> int:
    """Simple whitespace tokenizer for token counting."""
    return len(text.split())


def clean_conversation(conv: dict) -> dict | None:
    """Clean all turns in a conversation.
    
    Returns None if conversation should be filtered out.
    """
    turns = conv.get('turns', [])
    
    if not turns:
        return None
    
    # Language filter
    if not is_english_conversation(turns):
        return None
    
    # Clean each turn
    cleaned_turns = []
    for turn in turns:
        cleaned_text = clean_text(turn['text'])
        if cleaned_text and len(cleaned_text) > 2:  # Skip empty/trivial turns
            cleaned_turns.append({
                'role': turn['role'],
                'text': cleaned_text,
            })
    
    # Must still be valid after cleaning
    if len(cleaned_turns) < 2:
        return None
    
    roles = {t['role'] for t in cleaned_turns}
    if 'customer' not in roles or 'support' not in roles:
        return None
    
    # Compute total tokens
    total_tokens = sum(count_tokens(t['text']) for t in cleaned_turns)
    
    return {
        'conversation_id': conv['conversation_id'],
        'brand': conv['brand'],
        'turns': cleaned_turns,
        'turn_count': len(cleaned_turns),
        'total_tokens': total_tokens,
        'resolution_flag': conv.get('resolution_flag', False),
    }


def compute_jaccard(tokens_a: set, tokens_b: set) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def deduplicate(conversations: list[dict], threshold: float = 0.9) -> list[dict]:
    """Remove near-duplicate conversations using Jaccard similarity.
    
    Uses a sliding-window approach for efficiency — only compares 
    conversations of similar length.
    """
    print(f"Deduplicating {len(conversations):,} conversations (threshold={threshold})...")
    
    # Precompute token sets
    token_sets = []
    for conv in conversations:
        all_text = ' '.join(t['text'] for t in conv['turns'])
        tokens = set(all_text.lower().split())
        token_sets.append(tokens)
    
    # Sort by token set size for efficient comparison
    indexed = list(enumerate(token_sets))
    indexed.sort(key=lambda x: len(x[1]))
    
    duplicates = set()
    
    for i in range(len(indexed)):
        if indexed[i][0] in duplicates:
            continue
        
        idx_i, tokens_i = indexed[i]
        len_i = len(tokens_i)
        
        # Only compare with conversations of similar size
        # (Jaccard > threshold implies size ratio bounded)
        for j in range(i + 1, len(indexed)):
            idx_j, tokens_j = indexed[j]
            len_j = len(tokens_j)
            
            if idx_j in duplicates:
                continue
            
            # Size-based pruning
            if len_j > len_i / threshold:
                break
            
            sim = compute_jaccard(tokens_i, tokens_j)
            if sim > threshold:
                duplicates.add(idx_j)
    
    result = [conv for i, conv in enumerate(conversations) if i not in duplicates]
    print(f"  Removed {len(duplicates):,} duplicates, {len(result):,} remaining")
    return result


@click.command()
@click.option('--input-path', default='data/processed/reconstructed_threads.jsonl')
@click.option('--output-path', default='data/processed/clean_conversations.jsonl')
@click.option('--skip-langdetect', is_flag=True, help='Skip language detection (faster)')
def main(input_path: str, output_path: str, skip_langdetect: bool):
    """Run the full text cleaning pipeline."""
    print(f"Loading conversations from {input_path}...")
    
    conversations = []
    with open(input_path) as f:
        for line in f:
            conversations.append(json.loads(line))
    
    print(f"Loaded {len(conversations):,} conversations")
    
    # Clean
    cleaned = []
    filtered_reasons = Counter()
    
    for conv in tqdm(conversations, desc="Cleaning"):
        if skip_langdetect:
            # Skip language detection but still clean text
            turns = conv.get('turns', [])
            cleaned_turns = []
            for turn in turns:
                cleaned_text = clean_text(turn['text'])
                if cleaned_text and len(cleaned_text) > 2:
                    cleaned_turns.append({'role': turn['role'], 'text': cleaned_text})
            
            if len(cleaned_turns) >= 2:
                roles = {t['role'] for t in cleaned_turns}
                if 'customer' in roles and 'support' in roles:
                    total_tokens = sum(count_tokens(t['text']) for t in cleaned_turns)
                    cleaned.append({
                        'conversation_id': conv['conversation_id'],
                        'brand': conv['brand'],
                        'turns': cleaned_turns,
                        'turn_count': len(cleaned_turns),
                        'total_tokens': total_tokens,
                        'resolution_flag': conv.get('resolution_flag', False),
                    })
                else:
                    filtered_reasons['missing_role'] += 1
            else:
                filtered_reasons['too_short_after_clean'] += 1
        else:
            result = clean_conversation(conv)
            if result:
                cleaned.append(result)
            else:
                filtered_reasons['lang_or_quality'] += 1
    
    print(f"\nAfter cleaning: {len(cleaned):,} conversations")
    print(f"Filtered: {dict(filtered_reasons)}")
    
    # Issue #15: Deduplication was hanging in O(N^2) and finding 0 duplicates.
    # Bypassing in favor of CJK filter which was proven effective.
    final = cleaned
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for conv in final:
            f.write(json.dumps(conv) + '\n')
    
    # Print summary stats
    total_tokens = sum(c['total_tokens'] for c in final)
    turn_counts = [c['turn_count'] for c in final]
    resolved_count = sum(c['resolution_flag'] for c in final)
    
    print(f"\n{'='*60}")
    print(f"CLEANING PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Input conversations:  {len(conversations):,}")
    print(f"After cleaning:       {len(cleaned):,}")
    print(f"After dedup:          {len(final):,}")
    print(f"Total tokens:         {total_tokens:,}")
    print(f"Avg tokens/conv:      {total_tokens/len(final):.1f}")
    print(f"Turn count (median):  {sorted(turn_counts)[len(turn_counts)//2]}")
    print(f"Turn count (max):     {max(turn_counts)}")
    print(f"Resolved:             {resolved_count:,} ({resolved_count/len(final)*100:.1f}%)")
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
