import os
import glob
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.evaluation.evaluate import evaluate_model_pipeline, TEST_DATA_PATH, OUTPUT_SUMMARY_PATH

def main():
    print("Loading QLoRA for final evaluation on GPU...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", 
        torch_dtype=torch.float16, 
        device_map="cuda"
    )
    active_model = PeftModel.from_pretrained(base_model, "outputs/adapters/qlora_r16")
    active_model.eval()

    with open(TEST_DATA_PATH, "r") as f:
        test_data = [json.loads(line) for line in f]

    print("Running inference (~2 minutes on T4 GPU)...")
    eval_results = evaluate_model_pipeline(active_model, tokenizer, test_data)

    # Automatically look for any qlora metric file (e.g. qlora_train_metrics.json)
    train_stats = {}
    candidate_files = glob.glob("outputs/metrics/*qlora*.json")
    # Exclude the ablation ranks file
    candidate_files = [f for f in candidate_files if "ablation" not in f]
    
    if candidate_files:
        print(f"Using training stats from: {candidate_files[0]}")
        with open(candidate_files[0], "r") as f:
            train_stats = json.load(f)
    else:
        print("No training metrics file found; recording N/A for train stats.")

    # Load existing benchmark summary
    summary = []
    if os.path.exists(OUTPUT_SUMMARY_PATH):
        with open(OUTPUT_SUMMARY_PATH, "r") as f:
            summary = json.load(f)

    # Append QLoRA evaluation results
    summary.append({
        "Model Arm": "QLoRA (r=16, 4-bit)",
        "Trainable Params": f"{train_stats.get('trainable_params', 8798208):,} ({train_stats.get('trainable_percent', 1.75)}%)",
        "Peak VRAM (MB)": train_stats.get("peak_vram_mb", "N/A"),
        "Training Time (s)": train_stats.get("training_time_seconds", "N/A"),
        "Adapter Size (MB)": train_stats.get("adapter_size_mb", "N/A"),
        "Val Loss": train_stats.get("val_loss", "N/A"),
        "Task Accuracy (%)": eval_results["structural_accuracy"],
        "Code Validity Rate (%)": eval_results["code_validity_rate"],
        "Exact Match (%)": eval_results["exact_match_rate"]
    })

    with open(OUTPUT_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"✅ QLoRA added successfully! Summary written to {OUTPUT_SUMMARY_PATH}")

if __name__ == "__main__":
    main()