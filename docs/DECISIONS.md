# Decision Log

All non-obvious decisions are recorded here as they are made. This is a living document.

## Decision #1: Orchestration Approach
- **Date:** 2026-09-10
- **Decision:** Explicit Python functions, no heavy agent framework as orchestrator
- **Alternatives considered:** LangChain agent loops, LlamaIndex query engines, custom state machine
- **Why this one:** The "must explain and modify your own code live" rule makes framework-orchestrated control flow a liability. Explicit functions are traceable line-by-line.
- **What would change my mind:** If the pipeline required complex multi-step tool-use loops that are painful to implement manually

## Decision #2: Brand Selection — AmazonHelp
- **Date:** 2026-09-11
- **Decision:** Selected **AmazonHelp** as the target brand
- **Alternatives considered:** Top 10 brands by volume scored with deflection-trap metrics

| Brand | Conversations | Lex. Diversity | Self-Similarity | Resolution Rate | Composite |
|---|---|---|---|---|---|
| **AmazonHelp** | **81,092** | **0.6696** | **0.0227** | **0.3301** | **9.7289** |
| AmericanAir | 25,061 | 0.6628 | 0.0183 | 0.1860 | 6.7500 |
| Delta | 25,151 | 0.6896 | 0.0240 | 0.2008 | 5.7652 |
| SouthwestAir | 20,784 | 0.6648 | 0.0210 | 0.1362 | 4.3181 |
| TMobileHelp | 22,322 | 0.5914 | 0.0339 | 0.1773 | 3.0974 |
| SpotifyCares | 27,910 | 0.3839 | 0.0508 | 0.2363 | 1.7871 |
| AppleSupport | 76,639 | 0.4694 | 0.0655 | 0.1818 | 1.3022 |
| comcastcares | 23,442 | 0.3404 | 0.0654 | 0.1317 | 0.6857 |
| Ask_Spectrum | 17,770 | 0.3078 | 0.0734 | 0.1424 | 0.5969 |
| Uber_Support | 41,185 | 0.2676 | 0.1120 | 0.1596 | 0.3813 |

- **Why AmazonHelp:**
  1. Highest composite score (9.73) — nearly 1.5x second place
  2. Highest conversation volume (81K) — ample data for RAG retrieval
  3. Highest resolution proxy rate (33%) — substantive in-thread resolutions
  4. Very low self-similarity (0.023) — diverse, non-templated responses
  5. Manual verification: 0/20 sampled replies were deflections — all substantive
- **What would change my mind:** If manual reading revealed most "resolutions" are actually redirects to phone/chat (the resolution proxy can't catch that). Manual check showed this is not the case.

## Decision #3: Dataset Source
- **Date:** 2026-09-11
- **Decision:** Used TNE-AI/customer-support-on-twitter-conversation from Hugging Face (pre-reconstructed conversations with company labels)
- **Alternatives considered:** Raw thoughtvector/customer-support-on-twitter from Kaggle (3.98M individual tweets requiring manual thread reconstruction), gorkemsevinc raw version (only cleaned_text, no metadata)
- **Why this one:** Pre-reconstructed conversations save significant engineering time on thread linking. Contains 794K conversations across 109 brands with conversation_id, company, conversation text, and summary fields.
- **What would change my mind:** If the pre-reconstruction introduced errors in thread ordering or dropped important metadata. Spot-checking 5 random conversations showed correct ordering and alternating turns.
