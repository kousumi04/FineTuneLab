import os
import sys
import time
import yaml
import json
import argparse
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.models.peft_utils import load_model_and_tokenizer, apply_lora
from src.utils.memory_tracker import get_peak_vram_mb, reset_memory_stats

def get_directory_size_mb(directory_path: str) -> float:
    """Calculates total disk footprint of saved adapter weights in MB."""
    total_bytes = 0
    for dirpath, _, filenames in os.walk(directory_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_bytes += os.path.getsize(fp)
    return round(total_bytes / (1024 * 1024), 2)

def format_instruction(example):
    prompt = f"""### Instruction:
{example['instruction']}

### Code:
{example['input']['code']}

### Error:
{example['input']['error']}

### Response:
Cause: {example['output']['cause']}
Explanation: {example['output']['explanation']}
Fix: {example['output']['fix']}
Corrected Code:
{example['output']['corrected_code']}"""
    return {"text": prompt}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/qlora.yaml", help="Path to experiment YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"🚀 Starting Experiment: {config['experiment_name']} (4-bit: {config['use_4bit']})")

    dataset = load_dataset(
        "json",
        data_files={
            "train": "data/processed/train.jsonl",
            "validation": "data/processed/validation.jsonl"
        }
    )
    dataset = dataset.map(format_instruction)

    # Load Base Model (Quantized or Full FP16)
    model, tokenizer = load_model_and_tokenizer(
        model_name=config["model_name"],
        use_4bit=config["use_4bit"]
    )

    # Attach LoRA Adapters
    model = apply_lora(
        model=model,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"]
    )

    trainable_params, all_params = model.get_nb_trainable_parameters()
    trainable_pct = (trainable_params / all_params) * 100

    training_args = SFTConfig(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=float(config["learning_rate"]),
        num_train_epochs=config["num_train_epochs"],
        optim=config.get("optim", "adamw_torch"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=False,
        bf16=False,
        report_to="none",
        dataset_text_field="text",
        max_length=config.get("max_seq_length", 512)
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        args=training_args,
    )

    reset_memory_stats()
    start_time = time.time()
    train_result = trainer.train()
    training_time_seconds = round(time.time() - start_time, 2)
    peak_vram = round(get_peak_vram_mb(), 2)

    eval_metrics = trainer.evaluate()
    val_loss = round(eval_metrics.get("eval_loss", 0.0), 4)

    # Save artifact weights
    os.makedirs(config["output_dir"], exist_ok=True)
    trainer.model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    adapter_size_mb = get_directory_size_mb(config["output_dir"])

    metrics_payload = {
        "experiment_name": config["experiment_name"],
        "arm_type": "LoRA" if not config["use_4bit"] else "QLoRA",
        "lora_r": config["lora_r"],
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percent": round(trainable_pct, 4),
        "peak_vram_mb": peak_vram,
        "training_time_seconds": training_time_seconds,
        "adapter_size_mb": adapter_size_mb,
        "final_train_loss": round(train_result.metrics.get("train_loss", 0.0), 4),
        "val_loss": val_loss
    }

    os.makedirs("outputs/metrics", exist_ok=True)
    metrics_path = f"outputs/metrics/{config['experiment_name']}_train_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=4)

    print(f"✅ Run finished. Metrics logged to {metrics_path}")

if __name__ == "__main__":
    main()