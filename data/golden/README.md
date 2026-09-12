# Golden Evaluation Set - Sampling Notes

## Dataset Structure
This directory contains the 200 "Golden" examples used to calibrate and evaluate the Hiver Support Agent.
* `golden_calibration.jsonl` (60 examples): Strictly used for hyperparameter sweeping, threshold fitting, and prompt tuning.
* `golden_evaluation.jsonl` (140 examples): Strictly held-out test set for final benchmarking.
* `held_out_ids.json`: A list of all 200 IDs used by the FAISS indexer to explicitly exclude these conversations from the RAG knowledge base (preventing train-test leakage).

## Sampling Methodology
We did not use simple random sampling. To ensure the agent was tested against a rigorous and diverse set of challenges, we used **Stratified Clustering with Difficulty Oversampling**:
1. **Clustering:** Embedded 2,000 random conversations using `sentence-transformers` and clustered them into 10 semantic buckets via K-Means.
2. **Difficulty Tiering:** Ran a heuristic pass to classify conversations as "Hard" (e.g., short ambiguous queries, explicit anger, high turn counts) or "Easy".
3. **Stratified Draw:** Pulled examples proportionally from the 10 clusters, but artificially forced a **30% Hard / 70% Easy** ratio. This heavily penalizes naive models and stresses the escalation router.

## Labelling Notes
*Due to the constraints of the technical assignment (lack of human-in-the-loop time), the labels in this set were bootstrapped using a heuristic mock-labeller (`src/sampling/mock_manual_labels.py`).*

The labels provide:
* `intent`: The overarching goal (delivery_issue, refund_request, etc.)
* `grounding_available`: Boolean indicating if historical policy covers this.
* `escalate`: Boolean indicating if human intervention is required.
* `ideal_reply`: A baseline string of the perfect response.
