import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Union, List, Tuple

def load_model_and_tokenizer(
    model_name: str, 
    use_4bit: bool = False
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads the tokenizer and the base causal language model.
    Optionally applies 4-bit quantization for QLoRA.
    """
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Causal LMs usually don't have a padding token by default. 
    # We must set it for batched fine-tuning.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Important for causal LM training

    # Configure quantization if QLoRA is requested
    if use_4bit:
        print("Initializing 4-bit BitsAndBytes config for QLoRA...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",           # Optimal for normal-distributed weights
            bnb_4bit_use_double_quant=True,      # Quantize the quantization constants
            bnb_4bit_compute_dtype=torch.float16 # Compute happens in 16-bit
        )
    else:
        quantization_config = None

    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16, # Automatically maps model layers to GPU/CPU
    )

    # If using 4-bit, we must prepare the model (e.g., cast LayerNorms to fp32 for stability)
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def apply_lora(
    model: AutoModelForCausalLM,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Union[str, List[str]] = "all-linear"
) -> AutoModelForCausalLM:
    """
    Injects Trainable LoRA adapters into the frozen base model.
    
    Args:
        model: The base model (either standard or 4-bit quantized).
        r: The rank of the update matrices (lower = fewer params, less memory).
        lora_alpha: Scaling factor for the LoRA update. Usually 2 * r.
        lora_dropout: Dropout probability for adapter layers to prevent overfitting.
        target_modules: Which layers to attach adapters to.
    """
    print(f"Configuring LoRA (r={r}, alpha={lora_alpha}, targets={target_modules})...")
    
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # This function wraps the base model and replaces target layers with LoRA layers
    peft_model = get_peft_model(model, config)
    
    # This is a critical metric for your dashboard and interview!
    # It proves mathematically why PEFT is "Parameter Efficient"
    peft_model.print_trainable_parameters()
    
    return peft_model