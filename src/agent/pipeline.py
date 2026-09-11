"""Main Agent Pipeline Orchestrator.

Connects:
1. Intent Classification (k-NN)
2. OOD Detection (Centroid distance)
3. Retrieval (FAISS)
4. Context Budget Management
5. Escalation Routing (3-layer)
6. LLM Reply Drafting
"""
from __future__ import annotations

import json
from src.llm.provider_adapter import LLMProvider
from src.agent.retrieval_index import RetrievalIndex
from src.agent.intent_classifier import IntentClassifier
from src.agent.ood_detector import OODDetector
from src.agent.context_budget import ContextBudget, truncate_thread, truncate_retrievals
from src.agent.escalation import decide_escalation
from src.agent.draft_reply import draft_reply


def classify_and_respond(
    message: str,
    thread_history: list[dict],
    intent_classifier: IntentClassifier,
    ood_detector: OODDetector,
    retriever: RetrievalIndex,
    llm: LLMProvider,
    threshold_config: dict
) -> dict:
    """Run a message through the full agent pipeline."""
    
    # Track truncations for context budget
    truncation_log = {}
    
    # 1. Intent Classification
    intent = intent_classifier.predict(message)
    
    # 2. OOD Detection
    is_ood, distance = ood_detector.check_ood(message)
    
    # 3. Retrieval (Filter by intent)
    retrievals = retriever.search(message, top_k=3, intent_filter=intent)
    
    if not retrievals:
        # Fallback if intent filter yielded nothing
        retrievals = retriever.search(message, top_k=3)
        
    best_sim = retrievals[0]['similarity'] if retrievals else 0.0
    
    # 4. Context Budget Management
    budget = ContextBudget()
    
    formatted_thread = truncate_thread(thread_history, budget.thread_limit)
    if "[...older messages truncated...]" in formatted_thread:
        truncation_log['thread'] = True
        
    formatted_retrievals = truncate_retrievals(retrievals, budget.retrieval_limit)
    if any("[TRUNCATED]" in r for r in formatted_retrievals):
        truncation_log['retrieval'] = True
        
    # 5. Escalation Routing
    escalate, esc_reason, esc_layer = decide_escalation(
        customer_message=message,
        thread_history_str=formatted_thread,
        retrieval_similarity=best_sim,
        intent=intent,
        thresholds=threshold_config.get('per_intent', {}),
        global_threshold=threshold_config.get('global', 0.5),
        is_ood=is_ood,
        margin_width=threshold_config.get('margin_width', 0.05),
        llm=llm
    )
    
    # 6. Draft Reply (if not escalated)
    reply = ""
    if not escalate:
        reply = draft_reply(
            customer_message=message,
            thread_history_str=formatted_thread,
            retrievals=formatted_retrievals,
            intent=intent,
            llm=llm
        )
        
    # Format grounding sources for output
    grounding_sources = [r['customer_query'] for r in retrievals]
    
    return {
        'intent': intent,
        'draft_reply': reply,
        'grounding_sources': grounding_sources,
        'retrieval_confidence': best_sim,
        'escalate': escalate,
        'escalate_reason': f"{esc_layer}: {esc_reason}",
        'ood_flag': is_ood,
        'truncation_log': truncation_log
    }
