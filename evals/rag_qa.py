import evaluate
from datasets import load_dataset

from evals.utils import generate, peak_memory_gb

_squad_metric = evaluate.load("squad")

PROMPT_TEMPLATE = """Answer the question using only the information in the passage below. Be concise — one sentence or less.

Passage: {context}

Question: {question}

Answer:"""


def run(model, tokenizer, device, n_samples: int = 200) -> dict:
    dataset = load_dataset("rajpurkar/squad", split="validation")
    samples = list(dataset.select(range(n_samples)))

    predictions = []
    references = []
    total_tokens = 0
    total_time_s = 0.0
    total_ttft_s = 0.0

    for i, sample in enumerate(samples):
        prompt = PROMPT_TEMPLATE.format(
            context=sample["context"],
            question=sample["question"],
        )
        raw_output, n_tokens, tps, ttft = generate(model, tokenizer, device, prompt, max_new_tokens=64)
        pred_text = raw_output.strip().split("\n")[0].strip()

        predictions.append({"id": sample["id"], "prediction_text": pred_text})
        references.append({"id": sample["id"], "answers": sample["answers"]})

        total_tokens += n_tokens
        total_ttft_s += ttft
        if tps > 0:
            total_time_s += n_tokens / tps

        if (i + 1) % 10 == 0:
            print(f"  rag_qa: {i + 1}/{len(samples)}")

    squad_scores = _squad_metric.compute(predictions=predictions, references=references)
    n = len(samples)
    tps_overall = total_tokens / total_time_s if total_time_s > 0 else 0.0

    return {
        "task": "rag_qa",
        "n_samples": n,
        "exact_match": round(squad_scores["exact_match"], 4),
        "f1": round(squad_scores["f1"], 4),
        "tokens_per_sec": round(tps_overall, 2),
        "mean_ttft_ms": round(total_ttft_s / n * 1000, 1),
        "mean_output_tokens": round(total_tokens / n, 1),
        "peak_memory_gb": round(peak_memory_gb(), 3),
        "raw": [
            {
                "question": samples[i]["question"],
                "pred": predictions[i]["prediction_text"],
                "gold": references[i]["answers"]["text"],
            }
            for i in range(len(samples))
        ],
    }
