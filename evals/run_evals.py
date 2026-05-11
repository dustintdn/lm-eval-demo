"""
Usage:
    python -m evals.run_evals --model qwen --task all
    python -m evals.run_evals --model phi --task rag_qa --n_samples 50
"""
import argparse
import csv
import json
from pathlib import Path

from evals import classification, extraction, rag_qa
from evals.utils import load_model

TASKS = {
    "extraction": extraction,
    "rag_qa": rag_qa,
    "classification": classification,
}

DEFAULT_N = {
    "extraction": 150,
    "rag_qa": 200,
    "classification": 200,
}


def _update_summary(model_key: str, result: dict, load_time_s: float, disk_size_gb: float | None):
    summary_path = Path("results/summary.csv")
    task = result["task"]

    row = {
        "model": model_key,
        "model_load_time_s": load_time_s,
        "model_disk_size_gb": disk_size_gb,
        **{k: v for k, v in result.items() if k != "raw"},
    }

    existing = []
    if summary_path.exists():
        with open(summary_path, newline="") as f:
            reader = csv.DictReader(f)
            existing = [r for r in reader if not (r["model"] == model_key and r["task"] == task)]

    all_rows = existing + [row]
    fieldnames = list(dict.fromkeys(k for r in all_rows for k in r))

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Updated results/summary.csv")


def main():
    parser = argparse.ArgumentParser(description="Run LM evaluations")
    parser.add_argument("--model", required=True, choices=["qwen", "phi", "llama"])
    parser.add_argument("--task", required=True, choices=list(TASKS) + ["all"])
    parser.add_argument("--n_samples", type=int, default=None, help="Override sample count")
    args = parser.parse_args()

    model, tokenizer, device, load_time_s, disk_size_gb = load_model(args.model)

    tasks_to_run = list(TASKS) if args.task == "all" else [args.task]

    raw_dir = Path("results/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    for task_name in tasks_to_run:
        n = args.n_samples or DEFAULT_N[task_name]
        print(f"\n=== {task_name} | n={n} ===")

        result = TASKS[task_name].run(model, tokenizer, device, n_samples=n)

        out_path = raw_dir / f"{args.model}_{task_name}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved {out_path}")

        _update_summary(args.model, result, load_time_s, disk_size_gb)

        summary = {k: v for k, v in result.items() if k != "raw"}
        print(f"Results: {summary}")


if __name__ == "__main__":
    main()
