import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.evaluation.evaluate import evaluate_model_pipeline, TEST_DATA_PATH, OUTPUT_SUMMARY_PATH

def main():
    print("Loading QLoRA for final evaluation...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    # Load base model on CPU
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", 
        torch_dtype=torch.float32, 
        device_map="cpu"
    )
    active_model = PeftModel.from_pretrained(base_model, "outputs/adapters/qlora_r16")
    active_model.eval()

    with open(TEST_DATA_PATH, "r") as f:
        test_data = [json.loads(line) for line in f]

    print("Running inference... (This will take ~10-15 minutes on local CPU)")
    eval_results = evaluate_model_pipeline(active_model, tokenizer, test_data)

    # Load your local QLoRA train stats
    with open("outputs/metrics/qlora_r16_train_metrics.json", "r") as f:
        train_stats = json.load(f)

    # Append to the Colab summary
    with open(OUTPUT_SUMMARY_PATH, "r") as f:
        summary = json.load(f)

    summary.append({
        "Model Arm": "QLoRA (r=16, 4-bit)",
        "Trainable Params": f"{train_stats.get('trainable_params', 0):,} ({train_stats.get('trainable_percent', 0.0)}%)",
        "Peak VRAM (MB)": train_stats.get("peak_vram_mb", 0.0),
        "Training Time (s)": train_stats.get("training_time_seconds", 0.0),
        "Adapter Size (MB)": train_stats.get("adapter_size_mb", "N/A"),
        "Val Loss": train_stats.get("val_loss", "N/A"),
        "Task Accuracy (%)": eval_results["structural_accuracy"],
        "Code Validity Rate (%)": eval_results["code_validity_rate"],
        "Exact Match (%)": eval_results["exact_match_rate"]
    })

    with open(OUTPUT_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=4)
        
    print("✅ QLoRA added! The benchmark summary is now complete.")

if __name__ == "__main__":
    main()