import json
import random
import os
import ast

def validate_code(code_string: str) -> bool:
    """Validates if the provided string is syntactically valid Python code."""
    try:
        ast.parse(code_string)
        return True
    except SyntaxError:
        return False

def write_jsonl(data: list, filepath: str):
    """Writes a list of dictionaries to a JSONL file."""
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def main():
    raw_data_path = "data/raw/synthetic_dataset.json"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Load Data
    with open(raw_data_path, 'r') as f:
        data = json.load(f)
        
    # 2. Validate Ground Truth Code (Crucial for ML Data Quality)
    valid_data = []
    for idx, item in enumerate(data):
        target_code = item["output"]["corrected_code"]
        if validate_code(target_code):
            valid_data.append(item)
        else:
            print(f"Warning: Dropping example {idx} due to invalid syntax in corrected_code.")
            
    print(f"Retained {len(valid_data)}/{len(data)} valid examples.")

    # 3. Shuffle and Split (80/10/10)
    random.seed(42)
    random.shuffle(valid_data)
    
    total = len(valid_data)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)
    
    train_data = valid_data[:train_end]
    val_data = valid_data[train_end:val_end]
    test_data = valid_data[val_end:]
    
    # 4. Save to JSONL
    write_jsonl(train_data, f"{processed_dir}/train.jsonl")
    write_jsonl(val_data, f"{processed_dir}/validation.jsonl")
    write_jsonl(test_data, f"{processed_dir}/test.jsonl")
    
    print("✅ Dataset splitting complete:")
    print(f"  - Train: {len(train_data)} examples")
    print(f"  - Validation: {len(val_data)} examples")
    print(f"  - Test: {len(test_data)} examples")

if __name__ == "__main__":
    main()