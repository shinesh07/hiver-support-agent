# Sampling Notes

## Brand Selection Scores

| Brand | Lexical Diversity | Self-Similarity | Reply Length (median/IQR) | Resolution Proxy Rate | Composite Score |
|---|---|---|---|---|---|
| [TBD] | | | | | |

## Intent Taxonomy

[Finalized after manual review of golden-set sample — Phase 1.3 Step 4]

## Escalation Rubric v2

[Full rubric text — see Phase 4.5 of plan]

### Tier 1: Hard-Escalate (Deterministic)
[Rules 1.1 - 1.5 with positive/negative examples]

### Tier 2: Judgment-Tier (LLM-Decided)
[Rules 2.1 - 2.6 with anchoring scales]

### Tier 3: Auto-Handle
[Default criteria]

## Golden Set Split

- **Total examples:** ~200
- **Calibration slice:** 30% (~60), stratified across easy/hard
- **Evaluation slice:** 70% (~140), stratified across easy/hard
- **Random seed:** 42
- **Sampling method:** Stratified across k-means clusters × thread-length buckets, 30% hard-case oversampling

## Test-Retest Protocol

- Re-label 20% subsample after 24-48 hours
- Report: exact match on intent, exact match on escalate, Cohen's kappa on escalate
