import os
import time

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

MODEL_REGISTRY = {
    "qwen":  "Qwen/Qwen2.5-0.5B-Instruct",
    "phi":   "microsoft/Phi-3.5-mini-instruct",
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
}


class _FirstTokenTimer(StoppingCriteria):
    """Records wall time when the first new token is generated."""

    def __init__(self):
        self.ttft: float | None = None
        self._start: float = 0.0
        self._fired: bool = False

    def arm(self, start: float):
        self._start = start
        self._fired = False
        self.ttft = None

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if not self._fired:
            self.ttft = time.time() - self._start
            self._fired = True
        return False  # never stop generation early


_timer = _FirstTokenTimer()


def _model_disk_size_gb(model_id: str) -> float | None:
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = os.path.join(cache_dir, "models--" + model_id.replace("/", "--"))
    if not os.path.exists(model_dir):
        return None
    total_bytes = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, files in os.walk(model_dir)
        for f in files
    )
    return round(total_bytes / 1e9, 3)


def load_model(model_key: str):
    model_id = MODEL_REGISTRY[model_key]
    print(f"Loading {model_id}...")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model = model.to(device).eval()
    load_time_s = round(time.time() - t0, 2)

    disk_size_gb = _model_disk_size_gb(model_id)
    print(f"Loaded on {device} ({dtype}) | load: {load_time_s}s | disk: {disk_size_gb}GB")
    return model, tokenizer, device, load_time_s, disk_size_gb


def generate(model, tokenizer, device, prompt: str, max_new_tokens: int = 256):
    if tokenizer.chat_template:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = tokenizer(formatted, return_tensors="pt")["input_ids"].to(device)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    input_len = input_ids.shape[1]
    t0 = time.time()
    _timer.arm(t0)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([_timer]),
        )
    elapsed = time.time() - t0

    new_tokens = output_ids.shape[1] - input_len
    tps = new_tokens / elapsed if elapsed > 0 else 0.0
    ttft = _timer.ttft if _timer.ttft is not None else elapsed
    text = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return text, new_tokens, tps, ttft


def peak_memory_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return psutil.Process().memory_info().rss / 1e9
