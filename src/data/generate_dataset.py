import json
import random
import os
from typing import List, Dict, Any

# Ensure output directory exists
os.makedirs("data/raw", exist_ok=True)

def generate_type_error_examples(count: int) -> List[Dict[str, Any]]:
    """Generates TypeError examples (e.g., adding string and int)."""
    examples = []
    var_names1 = ["age", "count", "total", "index", "size"]
    var_names2 = ["name", "prefix", "suffix", "message", "label"]
    
    for _ in range(count):
        v1 = random.choice(var_names1)
        v2 = random.choice(var_names2)
        val1 = random.randint(1, 100)
        val2 = f'"{random.choice(["Item", "User", "Data", "Value"])}"'
        
        code = f"{v1} = {val1}\n{v2} = {val2}\nresult = {v1} + {v2}"
        error = f"TypeError: unsupported operand type(s) for +: 'int' and 'str'"
        
        examples.append({
            "instruction": "Explain the Python error and provide a fix.",
            "input": {"code": code, "error": error},
            "output": {
                "cause": f"You are trying to add an integer (`{v1}`) and a string (`{v2}`) together.",
                "explanation": "Python is strongly typed and does not automatically convert integers to strings during concatenation.",
                "fix": f"Convert the integer to a string using `str({v1})` before concatenation.",
                "corrected_code": f"{v1} = {val1}\n{v2} = {val2}\nresult = str({v1}) + {v2}"
            }
        })
    return examples

def generate_index_error_examples(count: int) -> List[Dict[str, Any]]:
    """Generates IndexError examples (e.g., accessing out-of-bounds list index)."""
    examples = []
    list_names = ["items", "users", "data", "records", "results"]
    
    for _ in range(count):
        lst = random.choice(list_names)
        length = random.randint(2, 5)
        bad_idx = length + random.randint(1, 3)
        list_str = "[" + ", ".join(str(random.randint(1, 10)) for _ in range(length)) + "]"
        
        code = f"{lst} = {list_str}\nprint({lst}[{bad_idx}])"
        error = "IndexError: list index out of range"
        
        examples.append({
            "instruction": "Explain the Python error and provide a fix.",
            "input": {"code": code, "error": error},
            "output": {
                "cause": f"You are trying to access index {bad_idx} in `{lst}`, which only has {length} elements (indices 0 to {length-1}).",
                "explanation": "Attempting to access an index that does not exist in a list raises an IndexError.",
                "fix": f"Ensure the index is within the valid range using `len({lst})`.",
                "corrected_code": f"{lst} = {list_str}\nif {bad_idx} < len({lst}):\n    print({lst}[{bad_idx}])\nelse:\n    print('Index out of bounds')"
            }
        })
    return examples

def generate_key_error_examples(count: int) -> List[Dict[str, Any]]:
    """Generates KeyError examples (e.g., missing dictionary key)."""
    examples = []
    dict_names = ["config", "user_data", "settings", "payload", "cache"]
    keys = ["id", "name", "status", "role", "email"]
    
    for _ in range(count):
        d_name = random.choice(dict_names)
        valid_key = random.choice(keys)
        bad_key = random.choice([k for k in keys if k != valid_key])
        
        code = f'{d_name} = {{"{valid_key}": "value"}}\nprint({d_name}["{bad_key}"])'
        error = f"KeyError: '{bad_key}'"
        
        examples.append({
            "instruction": "Explain the Python error and provide a fix.",
            "input": {"code": code, "error": error},
            "output": {
                "cause": f"The key `'{bad_key}'` does not exist in the dictionary `{d_name}`.",
                "explanation": "Accessing a dictionary directly with square brackets raises a KeyError if the key is missing.",
                "fix": f"Use the `.get()` method, which safely returns `None` (or a default value) if the key is not found.",
                "corrected_code": f'{d_name} = {{"{valid_key}": "value"}}\nprint({d_name}.get("{bad_key}", "Key not found"))'
            }
        })
    return examples

def main():
    random.seed(42) # Ensure deterministic generation
    dataset = []
    
    # Generate 500 of each type for 1,500 total examples
    dataset.extend(generate_type_error_examples(500))
    dataset.extend(generate_index_error_examples(500))
    dataset.extend(generate_key_error_examples(500))
    
    output_path = "data/raw/synthetic_dataset.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"✅ Generated {len(dataset)} examples and saved to {output_path}")

if __name__ == "__main__":
    main()