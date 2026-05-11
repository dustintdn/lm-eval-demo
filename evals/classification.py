import random
from datasets import load_dataset

from evals.utils import generate, peak_memory_gb

N_FEW_SHOT = 5
RANDOM_SEED = 42

PROMPT_TEMPLATE = """You are a customer intent classifier. Classify the customer query into exactly one of the intents listed below.
Respond with only the intent label, nothing else.

Intents:
{labels}

Examples:
{examples}

Query: "{query}"
Intent:"""


def _build_few_shot(train_dataset, labels: list[str], n: int) -> str:
    rng = random.Random(RANDOM_SEED)
    # One example per label (first occurrence), then sample n
    seen: dict[str, str] = {}
    for item in train_dataset:
        label = labels[item["label"]]
        if label not in seen:
            seen[label] = item["text"]
        if len(seen) == len(labels):
            break
    pool = list(seen.items())
    chosen = rng.sample(pool, min(n, len(pool)))
    return "\n".join(f'Query: "{text}"\nIntent: {label}' for label, text in chosen)


def _match(pred: str, gold: str) -> bool:
    return pred.strip().lower() == gold.strip().lower()


def run(model, tokenizer, device, n_samples: int = 200) -> dict:
    test_ds = load_dataset("PolyAI/banking77", split="test")
    train_ds = load_dataset("PolyAI/banking77", split="train")
    labels: list[str] = test_ds.features["label"].names

    labels_str = "\n".join(f"- {l}" for l in labels)
    few_shot_str = _build_few_shot(train_ds, labels, N_FEW_SHOT)

    samples = list(test_ds.select(range(n_samples)))
    results = []
    total_tokens = 0
    total_time_s = 0.0

    for i, sample in enumerate(samples):
        gold_label = labels[sample["label"]]
        prompt = PROMPT_TEMPLATE.format(
            labels=labels_str,
            examples=few_shot_str,
            query=sample["text"],
        )
        raw_output, n_tokens, tps = generate(model, tokenizer, device, prompt, max_new_tokens=16)
        pred_label = raw_output.strip().split("\n")[0].strip()
        correct = _match(pred_label, gold_label)

        results.append({
            "query": sample["text"],
            "gold": gold_label,
            "pred": pred_label,
            "correct": correct,
        })

        total_tokens += n_tokens
        if tps > 0:
            total_time_s += n_tokens / tps

        if (i + 1) % 10 == 0:
            print(f"  classification: {i + 1}/{len(samples)}")

    n = len(results)
    accuracy = sum(r["correct"] for r in results) / n
    tps_overall = total_tokens / total_time_s if total_time_s > 0 else 0.0

    return {
        "task": "classification",
        "n_samples": n,
        "accuracy": round(accuracy, 4),
        "tokens_per_sec": round(tps_overall, 2),
        "mean_output_tokens": round(total_tokens / n, 1),
        "peak_memory_gb": round(peak_memory_gb(), 3),
        "raw": results,
    }
