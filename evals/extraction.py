import json
import re
from pathlib import Path

from evals.utils import generate, peak_memory_gb

FIELDS = ["job_title", "company", "location", "salary_range", "years_experience_required"]

PROMPT_TEMPLATE = """Extract the following fields from the job posting as a JSON object.
Use null for any field that is not mentioned.

Fields to extract:
- job_title
- company
- location
- salary_range
- years_experience_required

Respond with only the JSON object and nothing else.

Job Posting:
{text}"""


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


def _normalize(v) -> str | None:
    if v is None:
        return None
    return str(v).lower().strip()


def _field_f1(pred: dict, gold: dict) -> dict:
    tp = fp = fn = 0
    per_field = {}
    for field in FIELDS:
        p_val = _normalize(pred.get(field))
        g_val = _normalize(gold.get(field))
        if g_val is None and p_val is None:
            per_field[field] = None  # not applicable
            continue
        elif g_val is None and p_val is not None:
            fp += 1
            per_field[field] = "fp"
        elif g_val is not None and p_val is None:
            fn += 1
            per_field[field] = "fn"
        elif p_val == g_val:
            tp += 1
            per_field[field] = "tp"
        else:
            fp += 1
            fn += 1
            per_field[field] = "wrong"

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "per_field": per_field}


def run(model, tokenizer, device, n_samples: int = 150) -> dict:
    data_path = Path(__file__).parent.parent / "data" / "job_postings.jsonl"
    samples = []
    with open(data_path) as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    samples = samples[:n_samples]

    results = []
    total_tokens = 0
    total_time_s = 0.0
    total_ttft_s = 0.0

    for i, sample in enumerate(samples):
        prompt = PROMPT_TEMPLATE.format(text=sample["text"])
        raw_output, n_tokens, tps, ttft = generate(model, tokenizer, device, prompt, max_new_tokens=128)
        pred = _parse_json(raw_output)
        metrics = _field_f1(pred, sample["labels"])

        results.append({
            "id": sample.get("id", i),
            "gold": sample["labels"],
            "pred": pred,
            "raw_output": raw_output,
            **{k: v for k, v in metrics.items() if k != "per_field"},
            "per_field": metrics["per_field"],
        })

        total_tokens += n_tokens
        total_ttft_s += ttft
        if tps > 0:
            total_time_s += n_tokens / tps

        if (i + 1) % 10 == 0:
            print(f"  extraction: {i + 1}/{len(samples)}")

    n = len(results)
    mean_f1 = sum(r["f1"] for r in results) / n
    mean_precision = sum(r["precision"] for r in results) / n
    mean_recall = sum(r["recall"] for r in results) / n
    tps_overall = total_tokens / total_time_s if total_time_s > 0 else 0.0

    return {
        "task": "extraction",
        "n_samples": n,
        "f1": round(mean_f1, 4),
        "precision": round(mean_precision, 4),
        "recall": round(mean_recall, 4),
        "tokens_per_sec": round(tps_overall, 2),
        "mean_ttft_ms": round(total_ttft_s / n * 1000, 1),
        "mean_output_tokens": round(total_tokens / n, 1),
        "peak_memory_gb": round(peak_memory_gb(), 3),
        "raw": results,
    }
