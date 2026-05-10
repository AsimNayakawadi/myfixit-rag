# Evaluation

This document describes the evaluation methodology, the test set, and the per-query results.

---

## Metrics

Three metrics are used, each probing a different aspect of system behaviour.

### Precision@K

Measures retrieval quality. For each query, we check whether each of the top-K retrieved chunks contains at least one keyword from a predefined relevance list. Precision@K is the proportion of the top-K that satisfy this condition.

We use **K = 3** for headline reporting.

### Grounding Rate

Measures generation faithfulness. For each query, we check whether the generated answer contains at least one keyword from the expected answer keyword list. The grounding rate is the proportion of queries where the answer is grounded.

This is a pragmatic proxy for whether the answer engages with the query topic, rather than producing generic filler text.

### Hallucination Rate

Measures the proportion of answers containing one or more pre-defined filler phrases that are not present in the retrieved context. Examples of monitored filler phrases:

- "reassemble in reverse order"
- "consult the manufacturer"
- "it is important to"
- "please refer to"
- "additional resources"

If the LLM produces any of these without them appearing in the retrieved context, the answer is flagged as hallucinated.

---

## Test Set

Seven queries spanning five device categories:

| # | Query | Category | Expected keywords |
|---|---|---|---|
| 1 | How do I remove the hard drive? | Mac | hard drive, screw, remove, cable |
| 2 | How do I replace the battery? | Mac / Phone | battery, screw, remove, replace |
| 3 | How do I remove the keyboard? | Mac | keyboard, screw, ribbon, remove |
| 4 | How do I replace an iPhone screen? | Phone | screen, display, remove, screw, cable |
| 5 | How do I replace a tablet battery? | Tablet | battery, remove, back, replace |
| 6 | How do I open a PlayStation 4? | Console | screw, cover, remove, open |
| 7 | How do I fix a camera lens? | Camera | lens, screw, remove, clean |

---

## Headline Results

| Metric | Score | Notes |
|---|---|---|
| Average Precision@3 | **0.87** | 87% of retrieved chunks relevant |
| Grounding Rate | **0.80** | 4 of 5 evaluated answers grounded |
| Hallucination Rate | **0.00** | No filler phrases detected |
| Indexed chunks | **53,482** | Full corpus, no subset |

---

## Per-Query Precision@3

| Query | Precision@3 |
|---|---|
| Hard drive | 0.67 |
| Battery | 1.00 |
| Keyboard | 0.67 |
| iPhone screen | 1.00 |
| Tablet battery | 1.00 |
| PlayStation 4 | 0.67 |
| Camera lens | 1.00 |
| **Average** | **0.87** |

See `figures/fig3_results.png` for the bar chart visualisation.

---

## Discussion

**Precision** — A Precision@3 of 0.87 indicates the structured chunking strategy (with Tools, Parts, and Verbs as additional retrieval signal) is working as intended. The chunks consistently surface steps from guides relevant to the query.

**Grounding** — The 0.80 grounding rate is encouraging for a system that uses no fine-tuning and a relatively small instruction-tuned model. The single ungrounded response in our test set was for the iPhone screen query, where the model produced a generic short response despite relevant context being present.

**Hallucination** — The zero hallucination rate is the most consequential safety result. The combination of strict system prompt rules, structural role separation, and deterministic decoding (temperature = 0.0) successfully prevented the model from emitting any of the monitored filler phrases.

This metric is a lower bound on actual hallucination quality — it cannot detect subtle factual errors. But within the scope of detectable filler phrases, the result is clean.

---

## Limitations and Future Evaluation

- Keyword-based metrics are conservative and miss subtle factual errors
- Test set is small (7 queries) — scaling to several hundred queries across all 15 categories is planned
- LLM-as-judge evaluation (using a stronger model to grade faithfulness on a calibrated scale) is the natural next step
- Human evaluation would provide the strongest signal but is expensive
