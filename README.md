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

## Quickstart

```bash
pip install -r requirements.txt

# run all tasks for a single model
python -m evals.run_evals --model qwen --task all

# run one task with a smaller sample count (faster)
python -m evals.run_evals --model phi --task rag_qa --n_samples 50
```

> **Note:** `meta-llama/Llama-3.2-3B-Instruct` requires accepting the license on
> HuggingFace before it can be downloaded.

Once results are populated, open `notebooks/analysis.ipynb` to explore.

---

## Hardware

Evaluations were run on <!-- e.g. "an M3 MacBook Pro (36GB unified memory)" -->.
Full pipeline (all 3 models × all 3 tasks) took approximately <!-- X hours -->.
