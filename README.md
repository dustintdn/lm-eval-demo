# SLM Evaluation

Benchmarking three open-weight small language models on tasks that reflect how SLMs are
actually deployed in production: structured extraction, retrieval-augmented Q&A, and intent
classification. The central question: **where do small models punch above their weight, and
where do they fall short?**

Results are committed to the repo — you can open the analysis notebook without re-running
inference.

---

## Models

| Model | Params | HuggingFace ID |
|---|---|---|
| Qwen2.5 | 0.5B | `Qwen/Qwen2.5-0.5B-Instruct` |
| Phi-3.5-mini | 3.8B | `microsoft/Phi-3.5-mini-instruct` |
| Llama-3.2 | 3B | `meta-llama/Llama-3.2-3B-Instruct` |

These three models span a 7× parameter range and represent distinct design philosophies.
Qwen2.5-0.5B is a genuine edge-deployment candidate; Phi-3.5-mini and Llama-3.2-3B compete
in the same weight class but are trained differently. No API keys required — all weights are
pulled from HuggingFace and run locally.

---

## Tasks

The tasks are chosen to reflect real SLM use cases rather than academic benchmarks.

### 1. Structured Extraction

**Dataset:** 150 synthetic job postings committed to `data/job_postings.jsonl`  
**Prompt:** Zero-shot. The model is given the full job posting text and asked to return a
JSON object with exactly five fields: `job_title`, `company`, `location`, `salary_range`,
`years_experience_required`. Fields not mentioned in the posting should be `null`.  
**Parsing:** Output is parsed as JSON directly; if that fails, a regex fallback extracts the
first `{...}` block. Parse failures return an empty dict (all fields missing).  
**Metrics:**
- **Field-level F1** — precision and recall computed over extracted key-value pairs per
  sample, averaged across the dataset. Each field is scored as TP (correct), FP (wrong
  value or hallucinated), or FN (missing).
- **JSON parse rate** — % of outputs that were valid JSON without the regex fallback.
  A model with high F1 but low parse rate is unreliable in production.

This task tests instruction-following, JSON output reliability, and handling of missing or
implicit fields.

### 2. RAG Q&A

**Dataset:** `rajpurkar/squad` v1.1, first 200 samples from the validation split  
**Prompt:** The model is given the passage and the question and asked to answer in one
sentence using only the provided passage — no open-domain recall.  
**Metrics:**
- **Exact match** — % of predictions that match any gold answer string exactly (after
  normalization)
- **Token-level F1** — token overlap between prediction and gold answers; standard SQuAD
  metric computed via HuggingFace `evaluate`

This task simulates retrieval-augmented generation. The context is provided, so the
metric is faithfulness and extraction quality, not world knowledge.

### 3. Intent Classification

**Dataset:** `PolyAI/banking77` test split, first 200 samples  
**Prompt:** Zero-shot with the full list of 77 banking-domain intent labels in the prompt,
plus 5 few-shot examples drawn from the training split (seeded, one example sampled per
label from a representative pool).  
**Metrics:**
- **Accuracy** — exact label match (case-insensitive)
- **Macro F1** — weights all 77 classes equally; penalizes models that are strong on
  common intents but fail on rare ones

Banking77 is dense (77 classes, many semantically similar) and commonly used in industry
benchmarks for intent routing. With the full label list in the prompt, this also stresses
long-context instruction-following.

---

## Measurements

Every eval run records the following per-model and per-sample:

| Metric | Description |
|---|---|
| `tokens_per_sec` | Decode throughput after the first token |
| `mean_ttft_ms` | Mean time to first token in ms (prefill cost) |
| `mean_output_tokens` | Average tokens generated per sample |
| `peak_memory_gb` | Peak memory (GPU: `max_memory_allocated`; CPU/MPS: process RSS) |
| `model_load_time_s` | Wall time to load weights from disk — cold-start cost |
| `model_disk_size_gb` | Size of model files in the HuggingFace cache |

All metrics flow into `results/summary.csv` (one row per model × task). The analysis
notebook derives additional efficiency ratios: accuracy per billion params, accuracy per
GB memory, accuracy per token/sec.

---

## Repo Structure

```
├── evals/
│   ├── run_evals.py        # CLI entrypoint (--model, --task, --n_samples)
│   ├── extraction.py       # structured extraction eval loop + field-level F1
│   ├── rag_qa.py           # SQuAD eval loop + exact match / F1
│   ├── classification.py   # Banking77 eval loop + accuracy
│   └── utils.py            # model loading, generation, timing helpers
├── data/
│   └── job_postings.jsonl  # committed synthetic dataset (150 job postings)
├── results/
│   ├── summary.csv         # one row per model × task, all metrics
│   └── raw/                # per-model JSON outputs — committed so the notebook runs offline
└── notebooks/
    └── analysis.ipynb      # plots, per-field breakdowns, confusion analysis, findings
```

---

## Running Evals

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Accept the Llama license (one-time)

`meta-llama/Llama-3.2-3B-Instruct` is gated. Visit the model page on HuggingFace,
accept the license, then authenticate locally:

```bash
huggingface-cli login
```

Qwen and Phi are ungated — no login required for those.

### 3. Run evals

Each command loads the model once and runs the specified task(s), writing results to
`results/raw/<model>_<task>.json` and updating `results/summary.csv`.

```bash
# Run all three tasks for a model (recommended — loads weights once)
python -m evals.run_evals --model qwen --task all
python -m evals.run_evals --model phi --task all
python -m evals.run_evals --model llama --task all

# Run a single task
python -m evals.run_evals --model qwen --task extraction
python -m evals.run_evals --model qwen --task rag_qa
python -m evals.run_evals --model qwen --task classification

# Reduce sample count for a quick smoke-test
python -m evals.run_evals --model qwen --task all --n_samples 10
```

**Valid `--model` values:** `qwen`, `phi`, `llama`  
**Valid `--task` values:** `extraction`, `rag_qa`, `classification`, `all`  
**Default sample counts:** extraction = 150, rag\_qa = 200, classification = 200

Re-running a model/task pair overwrites the existing raw JSON and updates that row in
`summary.csv` — it does not create duplicates.

### 4. Explore results

Once all nine model × task combinations have been run, open the analysis notebook:

```bash
jupyter lab notebooks/analysis.ipynb
```

---

## Key Findings

**Phi-3.5-mini is the strongest model overall, but carries a hidden prefill cost.** It nearly doubles Llama-3.2 on RAG Q&A (F1 69.1 vs 38.6) and leads on classification accuracy (74% vs 67%), but its mean time-to-first-token on classification is 2,210 ms — roughly 100× longer than Llama's 22 ms — because the large model hits a long prompt (77 labels + few-shot examples) hard during prefill. For real-time intent routing, that cost is prohibitive.

**Qwen-0.5B punches above its weight on extraction, then hits a capability floor.** It delivers extraction F1 within 11 points of Llama at 1/6th the parameters — a compelling edge-deployment profile. But classification accuracy drops to 15.5%, indicating very small models struggle with dense multi-class tasks that require reasoning over a long label list in context.

**Llama-3.2-3B offers the best balance for latency-sensitive deployments:** competitive accuracy across all three tasks, 22 ms TTFT on classification, and a moderate disk footprint relative to Phi.

**Salary range is the hardest extraction field across all three models**, suggesting numeric ranges with irregular formatting benefit from post-processing heuristics rather than pure LM extraction.

---

## Hardware

Evaluations were run on an M3 MacBook Pro (36GB unified memory).
Full pipeline (all 3 models × all 3 tasks) took approximately 1.5 hours.
