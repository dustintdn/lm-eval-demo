# SLM Evaluation

Benchmarking three open-weight small language models on tasks that reflect how SLMs are
actually used in production: structured extraction, retrieval-augmented Q&A, and intent
classification.

Results are committed to the repo — you can run the analysis notebook without re-running
inference.

---

## Models

| Model | Params | HuggingFace ID |
|---|---|---|
| Qwen | 0.5B | `Qwen/Qwen2.5-0.5B-Instruct` |
| Phi-3.5 | 3.8B | `microsoft/Phi-3.5-mini-instruct` |
| Llama-3.2 | 3B | `meta-llama/Llama-3.2-3B-Instruct` |

## Tasks

| Task | Dataset | Metric |
|---|---|---|
| Structured Extraction | Synthetic job postings | Field-level F1 |
| RAG Q&A | SQuAD v1.1 (passage provided) | Exact match + token F1 |
| Intent Classification | Banking77 (77 classes) | Accuracy |

---

## Key Findings

<!-- Fill in after running evals -->

---

## Repo Structure

```
evals/           eval loops for each task + CLI entrypoint
data/            committed synthetic dataset (job postings)
results/raw/     per-model JSON outputs — one file per model × task
results/         summary.csv — one row per model × task, all metrics
notebooks/       analysis.ipynb — plots and commentary
```

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

Re-running a model/task pair overwrites the existing raw JSON and updates that row
in `summary.csv` — it does not create duplicates.

### 4. Explore results

Once all nine model × task combinations have been run, open the analysis notebook:

```bash
jupyter lab notebooks/analysis.ipynb
```

---

## Hardware

Evaluations were run on <!-- e.g. "an M3 MacBook Pro (36GB unified memory)" -->.
Full pipeline (all 3 models × all 3 tasks) took approximately <!-- X hours -->.
