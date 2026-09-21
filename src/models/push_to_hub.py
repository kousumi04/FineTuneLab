import os
from transformers import AutoModelForCausalLM
from peft import PeftModel

def main():
    # Replace this with your actual Hugging Face username!
    hf_username = "Kousumi04"
    repo_name = f"{hf_username}/qwen2.5-0.5b-python-debugger-lora"
    
    base_model_id = "Qwen/Qwen2.5-0.5B"
    adapter_path = "outputs/adapters/qlora_r16"
    
    print(f"Loading base model {base_model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto"
    )
    
    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print(f"Pushing adapter to Hugging Face Hub at {repo_name}...")
    # This will create the repository and upload your adapter weights and config
    model.push_to_hub(repo_name)
    
    print(f"✅ Successfully uploaded to https://huggingface.co/{repo_name}")

if __name__ == "__main__":
    main()