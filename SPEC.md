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

### 1. Factual QA — TriviaQA
- **Dataset:** `trivia_qa` (rc.nocontext split), 200 samples from the validation set
- **Metric:** Exact match (normalized: lowercase, strip articles/punctuation)
- **Prompt format:** Zero-shot instruction prompt, no few-shot examples
- **Why:** Clean signal on world knowledge retrieval with no reasoning required

### 2. Math Reasoning — GSM8K
- **Dataset:** `gsm8k` (main split), 150 samples from the test set
- **Metric:** Exact match on final numeric answer (extracted from chain-of-thought output)
- **Prompt format:** Zero-shot chain-of-thought ("Think step by step.")
- **Why:** Tests multi-step reasoning; well-known benchmark with published baselines

### 3. Summarization — CNN/DailyMail
- **Dataset:** `cnn_dailymail` (3.0.0), 100 samples from the test set
- **Metric:** ROUGE-1, ROUGE-2, ROUGE-L
- **Prompt format:** "Summarize the following article in 3-5 sentences."
- **Why:** Open-ended generation task; ROUGE gives a comparable, if imperfect, signal

---

## Additional Measurements

For each model × task combination, also record:
- **Tokens/sec** — wall-clock throughput on the eval machine
- **Peak memory (GB)** — measured via `torch.cuda.max_memory_allocated` or `psutil` on CPU
- **Mean output length** — average tokens generated per sample

These go into a unified `results/summary.csv` so the notebook can plot accuracy vs. cost tradeoffs.

---

## Tooling

| Purpose | Tool |
|---|---|
| Eval harness | `lm-evaluation-harness` (EleutherAI) |
| Model loading | `transformers` + `torch` |
| Metrics | `rouge-score`, built-in harness metrics |
| Analysis | `pandas`, `matplotlib`, `seaborn` |
| Notebook | Jupyter |

Use `lm-evaluation-harness` for TriviaQA and GSM8K (standard tasks already implemented).
Write a lightweight custom eval loop for the summarization task (harness ROUGE support is limited).

---

## Repo Structure

```
lm-eval-demo/
├── SPEC.md
├── README.md               # written last — tells the story of the findings
├── requirements.txt
├── evals/
│   ├── run_evals.py        # CLI entrypoint: --model, --task, --n_samples
│   ├── summarization.py    # custom summarization eval loop
│   └── utils.py            # prompt formatting, answer extraction, timing helpers
├── results/
│   ├── summary.csv         # one row per model × task, all metrics
│   └── raw/                # per-model JSON outputs (committed)
│       ├── qwen_triviaqa.json
│       ├── phi_triviaqa.json
│       └── ...
└── notebooks/
    └── analysis.ipynb      # plots, commentary, findings
```

---

## Notebook Outline

1. **Setup** — load `results/summary.csv`
2. **Accuracy comparison** — grouped bar chart, model × task
3. **Throughput vs. accuracy scatter** — one point per model, per task; size = memory
4. **ROUGE breakdown** — R1 / R2 / RL side-by-side for summarization
5. **Key finding** — 1–2 sentence thesis backed by the data
6. **Limitations & next steps** — prompt sensitivity, contamination risk, what a bigger sweep would look like

---

## Constraints

- Eval subsets are capped (200 / 150 / 100 samples) so the full pipeline runs in under 2 hours on CPU
- All datasets loaded via HuggingFace `datasets` — no manual downloads
- Results JSON files committed so the notebook runs without re-running inference
- No proprietary models, no API keys

---

## Success Criteria

- [ ] All three models evaluated on all three tasks
- [ ] `results/summary.csv` populated and committed
- [ ] Notebook renders end-to-end without errors
- [ ] README has a clear thesis sentence and at least one surprising finding
- [ ] Repo is runnable by a stranger with `pip install -r requirements.txt`
