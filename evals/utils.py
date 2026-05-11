import time
import torch
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_REGISTRY = {
    "qwen":  "Qwen/Qwen2.5-0.5B-Instruct",
    "phi":   "microsoft/Phi-3.5-mini-instruct",
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
}


def load_model(model_key: str):
    model_id = MODEL_REGISTRY[model_key]
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(device).eval()
    print(f"Loaded on {device} ({dtype})")
    return model, tokenizer, device


def generate(model, tokenizer, device, prompt: str, max_new_tokens: int = 256):
    if tokenizer.chat_template:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    input_len = input_ids.shape[1]
    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    new_tokens = output_ids.shape[1] - input_len
    tps = new_tokens / elapsed if elapsed > 0 else 0
    text = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return text, new_tokens, tps


def peak_memory_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return psutil.Process().memory_info().rss / 1e9
