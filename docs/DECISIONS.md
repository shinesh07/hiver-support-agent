# Decision Log

All non-obvious decisions are recorded here as they are made. This is a living document.

## Decision #1: Orchestration Approach
- **Date:** [TBD]
- **Decision:** Explicit Python functions, no heavy agent framework as orchestrator
- **Alternatives considered:** LangChain agent loops, LlamaIndex query engines, custom state machine
- **Why this one:** The "must explain and modify your own code live" rule makes framework-orchestrated control flow a liability. Explicit functions are traceable line-by-line.
- **What would change my mind:** If the pipeline required complex multi-step tool-use loops that are painful to implement manually

## Decision #2: [Brand Selection]
- **Date:** [TBD — populate after Phase 1.1]
- **Decision:** [Selected brand]
- **Alternatives considered:** [Top candidates with deflection-trap scores]
- **Why this one:** [Quantitative + qualitative justification]
- **What would change my mind:** [Condition]

[Template for future entries below]

## Decision #N: [Title]
- **Date:** YYYY-MM-DD
- **Decision:**
- **Alternatives considered:**
- **Why this one:**
- **What would change my mind:**
