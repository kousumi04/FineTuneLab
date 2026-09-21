import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Union, List, Tuple

def load_model_and_tokenizer(
    model_name: str, 
    use_4bit: bool = False
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    # Force compute dtype to float16 for Colab T4 compatibility
    compute_dtype = torch.float16

    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",           
            bnb_4bit_use_double_quant=True,      
            bnb_4bit_compute_dtype=compute_dtype
        )
    else:
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto", 
        torch_dtype=compute_dtype,
    )

    # Force the model config to float16
    model.config.torch_dtype = compute_dtype

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
    
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )

    peft_model = get_peft_model(model, config)
    
    # ULTIMATE FIX: Force all trainable LoRA parameters to float32
    # This prevents the bfloat16 scaler crash on T4 GPUs
    for param in peft_model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    peft_model.print_trainable_parameters()
    
    return peft_model