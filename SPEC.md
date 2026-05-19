# LM Eval Demo — Project Spec

## Goal

A reproducible, portfolio-ready evaluation of three open-weight small language models
across three tasks. The central question: **where do SLMs punch above their weight,
and where do they fall short?**

Results are committed to the repo so reviewers can explore findings without running inference.

---

## Models

| Model | Params | Notes |
|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | Tiny baseline |
| `microsoft/Phi-3.5-mini-instruct` | 3.8B | Strong SLM, instruction-tuned |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | Mid-tier reference point |

All models pulled from HuggingFace and run locally. No API keys required.

---

## Tasks

These tasks are chosen to reflect how SLMs are actually deployed in production, not just
what appears in academic leaderboards.

### 1. Structured Extraction
- **Dataset:** 150 synthetic job postings (generated once, committed to repo as `data/job_postings.jsonl`)
- **Target fields:** `job_title`, `company`, `location`, `salary_range`, `years_experience_required`
- **Primary metric:** Field-level F1 — precision and recall over extracted key-value pairs, normalized
- **Reliability metric:** JSON parse rate — % of outputs that are valid JSON without the regex fallback. A model with good F1 but low parse rate is unreliable in production.
- **Prompt format:** "Extract the following fields as JSON: ..." — zero-shot, strict schema in prompt
- **Why:** The most common real-world SLM use case. Tests instruction-following, JSON output reliability, and ability to handle missing/implicit fields

### 2. RAG Q&A — SQuAD
- **Dataset:** `rajpurkar/squad` (v1.1), 200 samples from the validation set
- **Primary metric:** Exact match + token-level F1 (standard SQuAD metrics)
- **Reliability metric:** Faithfulness rate — % of predicted answers that appear verbatim in the source passage. Distinguishes "wrong but grounded" from "hallucinated."
- **Prompt format:** "Answer the question using only the provided passage. Question: {q} Passage: {p}"
- **Why:** Directly simulates retrieval-augmented generation — the model is given the context, so this tests faithfulness and extraction from a retrieved chunk, not open-domain recall

### 3. Intent Classification — Banking77
- **Dataset:** `PolyAI/banking77`, 200 samples from the test set (77 banking-domain intents)
- **Primary metric:** Accuracy (exact label match)
- **Supplementary metric:** Macro F1 — weights all 77 intent classes equally, penalizing models that are strong on common intents but fail on rare ones
- **Prompt format:** Zero-shot with full label list in prompt + 5 random few-shot examples from the training split
- **Why:** Intent routing is a high-volume, latency-sensitive SLM use case. Banking77 is dense (77 classes) and commonly used in industry benchmarks for this task

---

## Additional Measurements

### Per-sample (recorded inside each eval loop)

| Metric | Description |
|---|---|
| `tokens_per_sec` | Decode throughput — tokens generated per second after the first token |
| `mean_ttft_ms` | Mean time to first token in ms — prefill cost; high for long classification prompts |
| `mean_output_tokens` | Average new tokens generated per sample |
| `peak_memory_gb` | Peak memory usage (GPU: `max_memory_allocated`; CPU/MPS: process RSS) |

### Per-model (recorded once at load time)

| Metric | Description |
|---|---|
| `model_load_time_s` | Wall time to load weights from disk and move to device — cold-start cost |
| `model_disk_size_gb` | Size of model files in the HuggingFace cache — determines deployability |

### Derived (computed in the notebook from summary.csv)

| Metric | Formula | Why |
|---|---|---|
| Accuracy per billion params | `score / param_count_B` | Does bigger always mean better? |
| Accuracy per GB memory | `score / peak_memory_gb` | Quality ceiling for a given machine |
| Accuracy per token/sec | `score / tokens_per_sec` | Is the quality worth the latency cost? |

All per-sample and per-model metrics flow into `results/summary.csv` (one row per model × task).
Derived metrics are computed in the analysis notebook.

---

## Tooling

| Purpose | Tool |
|---|---|
| Model loading | `transformers` + `torch` |
| Datasets | `datasets` (HuggingFace) |
| Metrics | `evaluate` (HuggingFace) — SQuAD metric built-in |
| JSON parsing | `json` + regex fallback for extraction eval |
| Analysis | `pandas`, `matplotlib`, `seaborn` |
| Notebook | Jupyter |

All three tasks use a custom eval loop (no external harness dependency). This keeps the
code readable and makes the evaluation logic transparent to a reviewer.

---

## Repo Structure

```
lm-eval-demo/
├── SPEC.md
├── README.md                   # written last — tells the story of the findings
├── requirements.txt
├── data/
│   └── job_postings.jsonl      # committed synthetic dataset for extraction task
├── evals/
│   ├── run_evals.py            # CLI entrypoint: --model, --task, --n_samples
│   ├── extraction.py           # structured extraction eval loop + field-level F1
│   ├── rag_qa.py               # SQuAD eval loop + exact match / F1
│   ├── classification.py       # Banking77 eval loop + accuracy
│   └── utils.py                # model loading, prompt formatting, timing helpers
├── results/
│   ├── summary.csv             # one row per model × task, all metrics
│   └── raw/                    # per-model JSON outputs (committed)
│       ├── qwen_extraction.json
│       ├── qwen_rag_qa.json
│       ├── qwen_classification.json
│       └── ...
└── notebooks/
    └── analysis.ipynb          # plots, commentary, findings
```

---

## Notebook Outline

1. **Setup** — load `results/summary.csv`, define param counts and derived metrics
2. **Accuracy comparison** — grouped bar chart, model × task
3. **Efficiency Pareto** — scatter of tokens/sec vs. accuracy; bubble size = peak memory GB; one point per model × task
4. **Accuracy per billion params** — bar chart showing quality-per-parameter efficiency
5. **TTFT vs. task** — bar chart showing how prompt length drives time-to-first-token differently across tasks
6. **Extraction deep-dive** — per-field F1 breakdown; JSON parse rate per model
7. **Classification confusion** — hardest intents per model; accuracy vs. macro F1 gap
8. **Key finding** — 1–2 sentence thesis backed by the data
9. **Limitations & next steps** — prompt sensitivity, few-shot count sensitivity, quantization, what a bigger sweep would look like

---

## Constraints

- Eval subsets are capped (150 / 200 / 200 samples) so the full pipeline runs in under 2 hours on CPU
- All datasets loaded via HuggingFace `datasets` — no manual downloads
- Results JSON files committed so the notebook runs without re-running inference
- No proprietary models, no API keys

---

## Success Criteria

- [X] All three models evaluated on all three tasks
- [X] `results/summary.csv` populated and committed
- [X] Notebook renders end-to-end without errors
- [X] README has a clear thesis sentence and at least one surprising finding
- [X] Repo is runnable by a stranger with `pip install -r requirements.txt`
