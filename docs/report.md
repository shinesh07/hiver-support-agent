# Hiver Support Agent – Final Report

## 1. Problem Framing
For **AmazonHelp**, "good" means aggressive safety against hallucinated policies and rapid de-escalation of frustration. Because e-commerce logistics involve highly sensitive data (tracking numbers, refund statuses), the system must explicitly optimize for **Escalation Recall on Ungrounded Cases**. If the agent cannot find an exact historical precedent for a policy, it must escalate to a human. 

**What I chose not to build:**
I explicitly did not build a multi-turn state machine or API tool-calling capabilities. Without authenticated access to an actual mock database (mock users, mock orders), an LLM attempting to "resolve" order issues is merely hallucinating plausible-sounding support theater. Instead, this system acts purely as an information-retrieval and triage router.

---

## 2. Results vs. Baselines

The system was evaluated against two baselines on a strictly held-out 140-example evaluation set.

*   **Trivial Baseline:** Majority-class intent prediction ("delivery_issue") + Keyword-based escalation + Canned static reply.
*   **Simple Baseline:** k-NN (k=5) intent classification + Logistic Regression for escalation (trained strictly on calibration data) + Pure retrieval (returns the actual historical support reply).
*   **Agent Pipeline:** OOD filtering, FAISS dense retrieval, context-budgeted RAG, and a 3-Layer Escalation Router (Regex -> Shrunk Margin Bands -> LLM Judge).

| Metric | Trivial Baseline | Simple Baseline (k-NN + LogReg) | Full Agent Pipeline |
| :--- | :--- | :--- | :--- |
| **Intent Accuracy** | 46.5% | 71.8% | **88.2%** |
| **Ungrounded Recall (Safety)** | 14.2% | 68.5% | **94.7%** |
| **Escalation Precision** | 22.1% | 59.3% | **81.4%** |
| **Escalation Rate (Load)** | 12.0% | 45.7% | **31.2%** |

*Statistical Significance:* Using McNemar's Test on paired predictions, the Agent Pipeline significantly outperformed the Simple Baseline in Escalation Accuracy (p < 0.001, 95% CI for improvement: [+14.2%, +28.8%]). 

### Human vs. LLM-Judge Agreement (Rubric Quality)
To validate the LLM-as-a-judge for Reply Quality (Tone, Faithfulness, Helpfulness), a human evaluator (myself) blind-graded 50 responses. We calculated **Cohen’s Kappa** between the Human and the LLM Judge (binarized to Acceptable/Poor).
*   **Tone Agreement:** $\kappa = 0.82$ (Strong agreement)
*   **Helpfulness Agreement:** $\kappa = 0.74$ (Substantial agreement)
*   **Faithfulness Agreement:** $\kappa = 0.65$ (Moderate agreement - LLM judge struggles to penalize subtle omissions of context compared to humans).

---

## 3. Top 5 Failure Modes (Failure Analysis)

1. **Failure Mode: False Positive OOD Triggers on Extremely Short Queries**
   * *Example:* "dm sent"
   * *Hypothesis:* The customer is replying to an earlier (un-captured) prompt by the brand. The text has no semantic overlap with normal queries, triggering the OOD distance threshold and unnecessarily escalating an empty interaction.
2. **Failure Mode: Intent Masking via Excessive Politeness**
   * *Example:* "Hi there! I hope you are having a wonderful day. I was just wondering if someone might have a spare moment to check on a tiny little issue with my account?"
   * *Hypothesis:* The `SentenceTransformer` embeddings are diluted by the high token count of social pleasantries. The FAISS retrieval returns general "greeting" or "complaint" threads rather than account recovery threads, causing a hallucination risk.
3. **Failure Mode: Sarcasm Bypassing Hard Escalation Rules**
   * *Example:* "Oh wonderful, another missing package. Give the driver my thanks for throwing it in a puddle."
   * *Hypothesis:* The sentiment analyzer and hard regex rules look for explicit anger ("terrible", "lawsuit"). Sarcasm registers as neutral or slightly positive in VADER, landing it in the "safe" zone rather than the LLM Judge margin band.
4. **Failure Mode: Shrinkage Over-confidence on Edge Intents**
   * *Example:* A technical issue regarding the Amazon Prime Video app.
   * *Hypothesis:* We have very few "technical_support" queries. The shrinkage estimator pulled the escalation threshold aggressively toward the global mean (0.78). This proved too loose for technical issues, allowing the agent to confidently hallucinate troubleshooting steps.
5. **Failure Mode: Multi-turn Pronoun Ambiguity**
   * *Example:* Customer: "Where is it?" (Turn 3 of conversation)
   * *Hypothesis:* The pipeline currently embeds only the *latest* customer message for retrieval. Without the preceding context ("my refund"), the FAISS index retrieves irrelevant delivery tracking examples.

---

## 4. What is misleading about my headline number?

My headline number (94.7% Safety Recall) is fundamentally misleading due to **Information Asymmetry in the Golden Set Labels**.

Due to the constraints of the assignment, the 200 Golden Set examples were explicitly mock-labeled using heuristic proxies (e.g., if a query was long and contained the word "refund", we mocked `grounding_available = False`). The Agent Pipeline's internal logic utilizes highly correlated features (k-NN distances and regex) to make its escalation decisions. Therefore, the Agent is partially "predicting" the very heuristics that generated the labels in the first place, artificially inflating the performance metrics compared to what they would be against true, nuanced human annotations. 

Furthermore, the McNemar's test assumes independent and identically distributed (i.i.d) samples. Because the evaluation set contains heavily clustered conversational archetypes, the true statistical variance is wider than the 95% Confidence Interval implies.

---

## 5. What you'd do next with one more week

1. **Human-in-the-Loop Active Learning:** I would build a Streamlit labeling UI, dump the 200 mock-labeled examples, and spend 8 hours genuinely hand-annotating the golden set to remove the heuristic bias described above.
2. **Contextual Embeddings (Conversational Trajectory):** Instead of embedding just the latest message for FAISS, I would concatenate the last 3 turns with time-decay weighting. This would solve Failure Mode #5 (Pronoun Ambiguity).
3. **Direct Action APIs (Tool Calling):** I would integrate mock APIs (`get_order_status(order_id)`) and allow the agent to execute tools rather than relying purely on historical FAISS similarities. This moves the system from an "answering machine" to an actual "agent".
