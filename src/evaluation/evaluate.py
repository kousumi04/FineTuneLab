import json
import torch
import ast
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def extract_corrected_code(generated_text: str) -> str:
    """Extracts the code snippet appearing after the Corrected Code header."""
    marker = "Corrected Code:"
    if marker in generated_text:
        return generated_text.split(marker)[-1].strip()
    return ""

def validate_syntax(code_string: str) -> bool:
    """Uses Python's AST parser to check if the string is valid Python code."""
    if not code_string:
        return False
    try:
        ast.parse(code_string)
        return True
    except SyntaxError:
        return False

def main():
    base_model_id = "Qwen/Qwen2.5-0.5B"
    adapter_path = "outputs/adapters/qlora_r16"
    test_data_path = "data/processed/test.jsonl"
    report_path = "outputs/metrics/evaluation_report.txt"
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # Load base model in fp16
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Injecting trained LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # Load 10% held-out test data
    with open(test_data_path, "r") as f:
        test_data = [json.loads(line) for line in f]

    total = len(test_data)
    valid_structure_count = 0
    valid_syntax_count = 0
    exact_match_count = 0

    print(f"Starting evaluation on {total} test examples...")
    
    for item in tqdm(test_data, desc="Evaluating"):
        # Format the prompt exactly as it was during training, stopping at the Response header
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
                temperature=0.1 # Low temperature for deterministic code generation
            )
            
        # Decode and isolate the model's actual answer
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_only = generated_text.replace(prompt, "").strip()
        
        # Metric 1: Structural Integrity
        has_cause = "Cause:" in response_only
        has_exp = "Explanation:" in response_only
        has_fix = "Fix:" in response_only
        has_code = "Corrected Code:" in response_only
        
        if has_cause and has_exp and has_fix and has_code:
            valid_structure_count += 1
            
        # Metric 2: Code Validity
        extracted_code = extract_corrected_code(response_only)
        if validate_syntax(extracted_code):
            valid_syntax_count += 1
            
        # Metric 3: Exact Target Match
        target_code = item["output"]["corrected_code"].strip()
        if extracted_code == target_code:
            exact_match_count += 1

    # Calculate final percentages
    structure_acc = (valid_structure_count / total) * 100
    syntax_acc = (valid_syntax_count / total) * 100
    exact_match_acc = (exact_match_count / total) * 100

    report = f"""FineTuneLab - Evaluation Report
================================
Model: Qwen2.5-0.5B + LoRA (r=16)
Test Examples: {total}

Metrics:
--------
1. Structural Accuracy: {structure_acc:.2f}%
2. Code Syntax Validity: {syntax_acc:.2f}%
3. Code Exact Match: {exact_match_acc:.2f}%
"""

    with open(report_path, "w") as f:
        f.write(report)
        
    print("\n✅ Evaluation complete!")
    print(report)
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()