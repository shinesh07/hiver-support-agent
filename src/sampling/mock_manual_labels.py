"""Simulate manual labeling of the golden set for development purposes.

Generates plausible labels for:
- intent (e.g., delivery_issue, refund_request, product_inquiry, account_issue)
- escalate (bool)
- grounding_available (bool)
- ideal_reply (str)

Usage:
    python -m src.sampling.mock_manual_labels
"""
import json
import os
import random
import re

INTENTS = ['delivery_issue', 'refund_request', 'product_inquiry', 'account_issue', 'technical_support', 'complaint']

def generate_labels(c: dict) -> dict:
    all_text = ' '.join(t['text'].lower() for t in c['turns'])
    cust_text = ' '.join(t['text'].lower() for t in c['turns'] if t['role'] == 'customer')
    
    # Intent heuristics
    if 'refund' in cust_text or 'money back' in cust_text or 'charge' in cust_text:
        intent = 'refund_request'
    elif 'deliver' in cust_text or 'shipping' in cust_text or 'package' in cust_text or 'where is' in cust_text:
        intent = 'delivery_issue'
    elif 'password' in cust_text or 'login' in cust_text or 'account' in cust_text:
        intent = 'account_issue'
    elif 'how to' in cust_text or 'does it' in cust_text or 'product' in cust_text:
        intent = 'product_inquiry'
    elif 'broken' in cust_text or 'error' in cust_text or 'not working' in cust_text:
        intent = 'technical_support'
    else:
        intent = 'complaint' if c.get('difficulty_tier') == 'hard' else random.choice(INTENTS)
        
    # Escalate heuristics
    escalate = c.get('difficulty_tier') == 'hard' or 'manager' in cust_text or 'lawsuit' in cust_text
    
    # Grounding available: most easy things are grounded, hard things are often ungrounded
    if escalate:
        grounding_available = random.random() > 0.7  # 30% chance grounded
    else:
        grounding_available = random.random() > 0.2  # 80% chance grounded
        
    # Ideal reply mockup
    if escalate:
        ideal_reply = "I apologize for the immense frustration. Let me escalate this to our specialized team right away."
    else:
        ideal_reply = f"Thank you for reaching out regarding your {intent.replace('_', ' ')}. We're happy to help with this!"
        
    return {
        'intent': intent,
        'escalate': escalate,
        'grounding_available': grounding_available,
        'ideal_reply': ideal_reply
    }

def process_file(filepath: str):
    if not os.path.exists(filepath):
        return
        
    print(f"Labeling {filepath}...")
    labeled = []
    with open(filepath) as f:
        for line in f:
            c = json.loads(line)
            labels = generate_labels(c)
            c.update(labels)
            labeled.append(c)
            
    with open(filepath, 'w') as f:
        for c in labeled:
            f.write(json.dumps(c) + '\n')

def main():
    random.seed(42)
    process_file('data/golden/golden_calibration.jsonl')
    process_file('data/golden/golden_evaluation.jsonl')
    print("Golden sets have been mock-labeled successfully.")

if __name__ == '__main__':
    main()
