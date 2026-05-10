# Methodology

This document explains the design rationale and implementation details of the RAG pipeline. It complements the higher-level overview in the main README and the formal description in the paper.

---

## Pipeline Overview

The system is split into two phases. The offline indexing phase (Stages 1–7) is run once and persists the FAISS index and chunk DataFrame to disk. The query-time inference phase (Stages 8–10) runs for each user query.

| Stage | Function | Output |
|---|---|---|
| 1 | Load all 15 JSON files | Raw guides DataFrame |
| 2 | Flatten guides to steps | 53,482 step rows |
| 3 | Preprocess (text + lists) | Cleaned DataFrame |
| 4 | Build chunk_text | One chunk per step |
| 5 | Deduplicate | ~53k unique chunks |
| 6 | Embed chunks | (N, 384) matrix |
| 7 | FAISS index + persist | `faiss_index.bin` |
| 8 | Embed query, retrieve | Top-K chunks |
| 9 | Build context | Structured block |
| 10 | LLM generation | Grounded answer |

---

## Stage 1 — Dataset Loading

The MyFixit GitHub repository is cloned automatically on first run via `subprocess.run(["git", "clone", ...])`. All 15 JSON files in the `jsons/` subdirectory are loaded with `pd.read_json(path, lines=True)` and concatenated into a single DataFrame. A `source_file` column preserves the original device category for downstream filtering.

---

## Stage 2 — Step Extraction

Each guide row contains a nested `Steps` field — a list of dicts, one per repair step. We iterate through the DataFrame and explode this list into one row per step. Each output row carries:

- **Guide-level metadata**: `guidid`, `title`, `category`, `subject`, `url`
- **Step-level fields**: `step_order`, `step_id`, `text_raw`, `tools_annotated`, `tools_extracted`, `parts_clean`, `removal_verbs`, `images`

This produces ~53,482 step rows from ~2,224 guides.

---

## Stage 3 — Preprocessing

Preprocessing is intentionally light to preserve the original semantics:

- `normalize_text()` collapses whitespace and handles `None` values
- `clean_list_field()` strips `None`, empty strings, `'NA'`, and `'nan'` from list columns
- `extract_verb_names()` pulls unique verb names from the `removal_verbs` list-of-dicts
- `tools_final` prefers `tools_annotated` (human-curated) over `tools_extracted` (automatic) where both exist

---

## Stage 4 — Chunk Construction

The most consequential design decision in the pipeline. Each step is converted into a structured `chunk_text` string with explicit field labels:

```
Title: <guide title>
Subject: <component being repaired>
Step: <step number>
Tools: <tools used in this step>
Parts: <parts mentioned>
Verbs: <action verbs>
Instruction: <full step text>
```

**Why this format?**

1. **Title + Subject** anchor the device context. A query like "remove the hard drive" matches differently against a Mac guide vs. a smartphone guide — including these in the embedded text prevents cross-device confusion.
2. **Tools, Parts, Verbs** provide explicit lexical anchors for domain-specific entities. Action verbs like *remove*, *disconnect*, *lift* are strong signal for repair queries.
3. **Instruction** is the original text, preserved verbatim, that ultimately appears in the LLM context window.

Empty-instruction rows are dropped at this stage.

---

## Stage 5 — Deduplication

Exact string match deduplication on `chunk_text` via `df.drop_duplicates()`. We use exact match rather than near-duplicate detection because near-duplicate stepwise instructions often differ in small but meaningful ways (e.g., different tools or part names) — over-aggressive deduplication would erase that signal.

`SUBSET_SIZE = None` means the full corpus (~53,482 unique chunks) is indexed.

---

## Stage 6 — Embedding

Model: `sentence-transformers/all-MiniLM-L6-v2`

- 384-dimensional dense vectors
- ~90 MB on disk
- Trained for semantic similarity (NLI + STSb)
- Fast on CPU; very fast on MPS / CUDA

Each chunk's `chunk_text` is encoded with `encoder.encode(texts, convert_to_numpy=True, show_progress_bar=True)`. The same encoder is loaded at query time so chunk and query vectors live in the same embedding space.

**Device selection** follows this priority chain:
```python
if torch.backends.mps.is_available():    DEVICE = "mps"
elif torch.cuda.is_available():           DEVICE = "cuda"
else:                                     DEVICE = "cpu"
```

On an M-series MacBook Air with MPS, encoding ~53k chunks completes in roughly 3–4 minutes.

---

## Stage 7 — FAISS Indexing

We use `faiss.IndexFlatIP` with L2-normalised vectors. After L2 normalisation, inner product is mathematically equivalent to cosine similarity, and `IndexFlatIP` performs exact (non-approximate) search.

**Why IndexFlatIP and not IVF/HNSW?**
- The corpus is ~53k vectors at 384 dims — well within exact-search territory
- Search latency is sub-millisecond on CPU
- No recall loss compared to approximate methods
- No tuning required (no `nprobe`, no `ef_construction`)

Persistence is critical for practical use:
```python
faiss.write_index(index, "faiss_index.bin")
rag_df[SAVE_COLS].to_parquet("rag_chunks.parquet", index=False)
```

On subsequent sessions, both files are loaded directly and the embedding step is skipped entirely.

---

## Stage 8 — Retrieval

```python
def retrieve(query, encoder, index, k=5):
    vec = encoder.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(vec)
    scores, indices = index.search(vec, k)
    return scores[0], indices[0]
```

`TOP_K = 5` chunks are retrieved. `MAX_CONTEXT = 4` chunks are forwarded to the LLM. The slight asymmetry (retrieve 5, use 4) leaves margin for future filtering or re-ranking.

---

## Stage 9 — Context Construction

Retrieved chunks are assembled into a numbered list of structured source entries. Each entry includes:

- Source rank
- Cosine similarity score
- Guide title
- Device subject
- Step number
- Cleaned instruction text
- Source URL (included for traceability, but stripped from output by system prompt rule 4)

---

## Stage 10 — Generation

**Model**: Mistral 7B via Ollama (local inference)

**Why Ollama and not HuggingFace Transformers?**
- No API key required
- MPS-accelerated on Apple Silicon out of the box
- Cleaner separation between system / user roles in the chat API
- Suitable for privacy-sensitive industrial deployment

**Generation parameters**:
- `temperature = 0.0` — fully deterministic decoding, no sampling variance
- `repeat_penalty = 1.15` — suppresses phrase looping (a known failure mode of smaller instruction-tuned models)

The system prompt is passed in the `system` role; the assembled context plus user query are passed in the `user` role. See [PROMPT_DESIGN.md](PROMPT_DESIGN.md) for the full system prompt and injection mitigation strategy.
