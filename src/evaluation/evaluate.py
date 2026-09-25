import os
import ast
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B"
TEST_DATA_PATH = "data/processed/test.jsonl"
OUTPUT_SUMMARY_PATH = "outputs/metrics/benchmark_summary.json"

def validate_syntax(code_str: str) -> bool:
    if not code_str.strip():
        return False
    try:
        ast.parse(code_str)
        return True
    except SyntaxError:
        return False

def extract_corrected_code(response_text: str) -> str:
    marker = "Corrected Code:"
    if marker in response_text:
        return response_text.split(marker)[-1].strip()
    return ""

def evaluate_model_pipeline(model, tokenizer, test_data):
    total = len(test_data)
    valid_structure = 0
    valid_syntax = 0
    exact_matches = 0

    for item in tqdm(test_data, desc="Running Inference"):
        prompt = (
            f"### Instruction:\n{item['instruction']}\n\n"
            f"### Code:\n{item['input']['code']}\n\n"
            f"### Error:\n{item['input']['error']}\n\n"
            f"### Response:\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.1
            )
        raw_text = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

        # 1. Structural Correctness
        if all(k in raw_text for k in ["Cause:", "Explanation:", "Fix:", "Corrected Code:"]):
            valid_structure += 1

        # 2. Code Validity via AST
        extracted_code = extract_corrected_code(raw_text)
        if validate_syntax(extracted_code):
            valid_syntax += 1

        # 3. Exact Code Match
        target_code = item["output"]["corrected_code"].strip()
        if extracted_code == target_code:
            exact_matches += 1

    return {
        "structural_accuracy": round((valid_structure / total) * 100, 2),
        "code_validity_rate": round((valid_syntax / total) * 100, 2),
        "exact_match_rate": round((exact_matches / total) * 100, 2)
    }

def main():
    os.makedirs(os.path.dirname(OUTPUT_SUMMARY_PATH), exist_ok=True)
    with open(TEST_DATA_PATH, "r") as f:
        test_data = [json.loads(line) for line in f]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load Raw Base Model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
        device_map=device
    )

    runs = [
        {"name": "Base Model (Zero-Shot)", "type": "base", "adapter": None, "train_metric": None},
        {"name": "LoRA (r=16, FP16)", "type": "lora", "adapter": "outputs/adapters/lora_r16", "train_metric": "outputs/metrics/lora_r16_train_metrics.json"},
        {"name": "QLoRA (r=16, 4-bit)", "type": "qlora", "adapter": "outputs/adapters/qlora_r16", "train_metric": "outputs/metrics/qlora_r16_train_metrics.json"},
    ]

    summary_records = []

    for run in runs:
        print(f"\n--- Evaluating {run['name']} ---")
        if run["type"] == "base":
            active_model = base_model
        else:
            if not os.path.exists(run["adapter"]):
                print(f"Skipping {run['name']}; adapter not found at {run['adapter']}")
                continue
            active_model = PeftModel.from_pretrained(base_model, run["adapter"])

        active_model.eval()
        eval_results = evaluate_model_pipeline(active_model, tokenizer, test_data)

        # Pull logged training metrics if available
        train_stats = {}
        if run["train_metric"] and os.path.exists(run["train_metric"]):
            with open(run["train_metric"], "r") as f:
                train_stats = json.load(f)

        summary_records.append({
            "Model Arm": run["name"],
            "Trainable Params": f"{train_stats.get('trainable_params', 0):,} ({train_stats.get('trainable_percent', 0.0)}%)" if run["type"] != "base" else "0 (0.0%)",
            "Peak VRAM (MB)": train_stats.get("peak_vram_mb", 0.0) if run["type"] != "base" else "N/A",
            "Training Time (s)": train_stats.get("training_time_seconds", 0.0) if run["type"] != "base" else "N/A",
            "Adapter Size (MB)": train_stats.get("adapter_size_mb", "N/A"),
            "Val Loss": train_stats.get("val_loss", "N/A"),
            "Task Accuracy (%)": eval_results["structural_accuracy"],
            "Code Validity Rate (%)": eval_results["code_validity_rate"],
            "Exact Match (%)": eval_results["exact_match_rate"]
        })

    with open(OUTPUT_SUMMARY_PATH, "w") as f:
        json.dump(summary_records, f, indent=4)

    print(f"\n✅ All arms evaluated. Summary saved to {OUTPUT_SUMMARY_PATH}")

if __name__ == "__main__":
    main()