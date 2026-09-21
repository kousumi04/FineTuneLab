import os
import time
import yaml
import json
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig

import sys
# Ensure Python can find our custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.peft_utils import load_model_and_tokenizer, apply_lora
from src.utils.memory_tracker import get_peak_vram_mb, reset_memory_stats

def format_instruction(example):
    """Formats the JSONL dictionary into a structured text prompt for the model."""
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
    # 1. Load Configuration
    config_path = "configs/qlora.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    print(f"🚀 Starting Experiment: {config['experiment_name']}")

    # 2. Load Dataset
    print("Loading datasets...")
    dataset = load_dataset(
        "json", 
        data_files={
            "train": "data/processed/train.jsonl",
            "validation": "data/processed/validation.jsonl"
        }
    )
    
    # Map the formatting function across the dataset
    dataset = dataset.map(format_instruction)

    # 3. Load Model, Tokenizer, and inject LoRA
    model, tokenizer = load_model_and_tokenizer(
        model_name=config["model_name"], 
        use_4bit=config["use_4bit"]
    )
    
    model = apply_lora(
        model=model,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"]
    )

    # 4. Setup Training Arguments
    training_args = SFTConfig(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=float(config["learning_rate"]),
        num_train_epochs=config["num_train_epochs"],
        optim=config["optim"],                      
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        fp16=True,                                  
        report_to="none",                            
        dataset_text_field="text",                          max_length=config["max_seq_length"]     
    )

    # 5. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        args=training_args,
    )

    # 6. Train and track hardware metrics
    print("Starting training loop...")
    reset_memory_stats()
    start_time = time.time()
    
    train_result = trainer.train()
    
    end_time = time.time()
    peak_vram = get_peak_vram_mb()
    training_time_seconds = end_time - start_time

    # 7. Save adapter and metrics
    print(f"✅ Training complete in {training_time_seconds:.2f} seconds.")
    print(f"Peak VRAM used: {peak_vram:.2f} MB")
    
    trainer.model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    
    # Save metrics to JSON for the dashboard later
    os.makedirs("outputs/metrics", exist_ok=True)
    metrics = {
        "experiment_name": config["experiment_name"],
        "training_time_seconds": training_time_seconds,
        "peak_vram_mb": peak_vram,
        "train_loss": train_result.metrics.get("train_loss"),
        "lora_r": config["lora_r"]
    }
    
    with open(f"outputs/metrics/{config['experiment_name']}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()