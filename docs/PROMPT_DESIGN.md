# Prompt Design and Injection Mitigation

This document covers the system prompt used by the LLM generator and the strategies employed to prevent prompt injection attacks.

---

## System Prompt

The system prompt is passed in the `system` role of the Ollama chat API. It encodes eight explicit rules that together define the assistant's behaviour:

```
You are a precise repair assistant. You answer device repair questions
strictly using the provided context.

STRICT RULES — follow every one:

1. Use ONLY information from the context. Do not add steps, tips, or
   knowledge from outside the context.

2. Format your answer as a numbered list. Each step must reference its
   source like this: [Guide: <title>, Step <number>]

3. List the required tools at the top under "Tools needed:" before the
   steps. Only list tools that appear in the context.

4. Never include URLs, links, or web addresses in your answer.

5. Never say things like "reassemble in reverse order" or "consult
   additional resources" unless that exact phrase appears in the context.

6. If the context genuinely has zero relevant information for the
   question, respond with exactly:
   "No relevant repair steps were found in the dataset."

7. Never contradict yourself between any introductory assessment and the
   listed steps.

8. Do not pad the answer beyond what the context supports.
```

### Why negative formulations?

In iterative development with Mistral 7B, we observed that prohibitions ("Do not...", "Never...") are followed more reliably than permissions. We use the negative form wherever the rule is genuinely about preventing a behaviour, and the positive form only where prescribing structure (e.g., the citation format).

---

## Generation Parameters

| Parameter | Value | Rationale |
|---|---|---|
| `temperature` | 0.0 | Fully deterministic — no sampling variance |
| `repeat_penalty` | 1.15 | Suppresses phrase looping (a known small-model failure) |
| `num_ctx` | 8192 | Mistral 7B's context window — enough for top-4 chunks |

---

## Prompt Injection Mitigation

A RAG system passes externally retrieved text to an LLM, which in principle makes it a target for prompt injection. For the MyFixit corpus this risk is low because the dataset is curated, but the mitigations generalise to less-trusted corpora and are worth implementing as a matter of good practice.

### 1. Structural Separation of Roles

The system prompt is sent in the `system` message role; the retrieved context and user query are sent together in the `user` role:

```python
ollama.chat(
    model="mistral",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ],
    options={"temperature": 0.0, "repeat_penalty": 1.15}
)
```

Modern instruction-tuned models are trained to treat these roles differently. The system role takes precedence, making it considerably harder for an injected instruction in the user message to override the system prompt.

### 2. Labelled Context Structure

The retrieved context is presented as a numbered list of source entries with explicit field labels:

```
[Source 1 | Similarity: 0.87]
Title:       PowerBook G3 Hard Drive Replacement
Subject:     Hard Drive
Step:        7
Instruction: Remove the three screws securing the hard drive bracket...

---

[Source 2 | Similarity: 0.84]
...
```

The model encounters retrieved content inside this structured envelope rather than as free-form text. This provides an additional implicit cue that the content is data rather than instruction.

### 3. Deterministic Decoding

With `temperature = 0.0`, the model always selects the highest-probability next token. This removes the sampling variance that some prompt injection attacks rely on to occasionally produce attacker-favoured outputs.

---

## Limitations of the Mitigation Strategy

These mitigations substantially raise the cost of a successful prompt injection attack but are not sufficient against a determined adversary. Stronger defences would include:

- **Input sanitisation**: scanning retrieved text for instruction-like patterns and refusing to forward suspicious chunks
- **Output validation**: post-hoc checks that the answer cites valid source IDs and contains no URLs
- **Adversarial testing**: a regular test suite of known injection payloads run against the pipeline

These are tracked as future work.

---

## Example: System in Action

**Query**: How do I remove the keyboard?

**Generated answer**:
```
Tools needed: Phillips screwdriver, spudger

1. Pull the keyboard release tabs toward you and lift up on the keyboard
   until it pops free.
   [Guide: PowerBook G3 Pismo Keyboard Replacement, Step 1]

2. Lift the keyboard out by pulling it up and away from you. Rest the
   keyboard face down on top of the trackpad.
   [Guide: PowerBook G3 Pismo Keyboard Replacement, Step 2]

3. Disconnect the keyboard ribbon cable from the logic board by gently
   pulling on the connector.
   [Guide: PowerBook G3 Pismo Keyboard Replacement, Step 3]
```

The answer cites every step with `[Guide: ..., Step n]`, lists tools at the top, includes no URLs, contains no filler phrases, and uses only information from retrieved chunks — all as specified by the system prompt.
