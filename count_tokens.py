import json
from transformers import AutoTokenizer

def count_tokens_in_jsonl(file_path, model_name="google/gemma-2-2b"):
    """
    Count the total number of tokens in a JSONL file using the specified tokenizer.

    Args:
        file_path (str): Path to the JSONL file.
        model_name (str): Model name to load the tokenizer.

    Returns:
        int: Total number of tokens in the file.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    total_tokens = 0

    # Read the JSONL file line by line
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse the JSON object
            json_obj = json.loads(line)
            
            # Extract the "text" field (or modify if the text field is named differently)
            text = json_obj.get("text", "")
            
            # Tokenize the text and count tokens
            tokens = tokenizer.encode(text, add_special_tokens=True)
            total_tokens += len(tokens)

    return total_tokens


if __name__ == "__main__":
    # Path to your JSONL file
    jsonl_file_path = "/lfs/skampere1/0/iddah/selected_data/humaneval/top_p0.009.jsonl"
    
    # Count tokens
    total_tokens = count_tokens_in_jsonl(jsonl_file_path)

    print(f"Total number of tokens in the file: {total_tokens}")
