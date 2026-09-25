import os
import sys
import json
import time
import torch
import gc
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import SFTTrainer, SFTConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.models.peft_utils import load_model_and_tokenizer, apply_lora
from src.utils.memory_tracker import get_peak_vram_mb, reset_memory_stats

# ONLY RUN THE REMAINING RANK
RANKS = [32] 
ABLATION_METRICS_PATH = "outputs/metrics/ablation_ranks.json"
TEST_DATA_PATH = "data/processed/test.jsonl"

def format_instruction(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Code:\n{example['input']['code']}\n\n### Error:\n{example['input']['error']}\n\n### Response:\nCause: {example['output']['cause']}\nExplanation: {example['output']['explanation']}\nFix: {example['output']['fix']}\nCorrected Code:\n{example['output']['corrected_code']}"
    }

def main():
    dataset = load_dataset("json", data_files={"train": "data/processed/train.jsonl", "validation": "data/processed/validation.jsonl"}).map(format_instruction)
    
    # Load existing records so we don't overwrite Rank 4, 8, and 16
    ablation_records = []
    if os.path.exists(ABLATION_METRICS_PATH):
        with open(ABLATION_METRICS_PATH, "r") as f:
            try:
                ablation_records = json.load(f)
            except json.JSONDecodeError:
                pass

    for r in RANKS:
        alpha = r * 2
        exp_dir = f"outputs/adapters/ablation_qlora_r{r}"
        print(f"\n================ Running Ablation: Rank {r} (Alpha {alpha}) ================")

        model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-0.5B", use_4bit=True)
        model = apply_lora(model, r=r, lora_alpha=alpha, lora_dropout=0.05, target_modules="all-linear")

        trainable_params, all_params = model.get_nb_trainable_parameters()

        training_args = SFTConfig(
            output_dir=exp_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=2,
            optim="adamw_torch",
            logging_steps=20,
            eval_strategy="epoch",
            save_strategy="no",
            fp16=False,
            bf16=False,
            report_to="none",
            dataset_text_field="text",
            max_length=384
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            processing_class=tokenizer,
            args=training_args
        )

        reset_memory_stats()
        t0 = time.time()
        trainer.train()
        train_time = round(time.time() - t0, 2)
        peak_vram = round(get_peak_vram_mb(), 2)

        eval_res = trainer.evaluate()
        val_loss = round(eval_res.get("eval_loss", 0.0), 4)

        # Skip inference to prevent VRAM Out-of-Memory lock
        task_acc = 0.0

        ablation_records.append({
            "rank": r,
            "trainable_params": trainable_params,
            "trainable_percent": round((trainable_params / all_params) * 100, 4),
            "peak_vram_mb": peak_vram,
            "training_time_seconds": train_time,
            "val_loss": val_loss,
            "task_accuracy": task_acc
        })

        # Save after EVERY rank just in case it crashes again
        os.makedirs(os.path.dirname(ABLATION_METRICS_PATH), exist_ok=True)
        with open(ABLATION_METRICS_PATH, "w") as f:
            json.dump(ablation_records, f, indent=4)

        # Force aggressive garbage collection before emptying cache
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n✅ Rank ablation complete. Results saved to {ABLATION_METRICS_PATH}")

if __name__ == "__main__":
    main()