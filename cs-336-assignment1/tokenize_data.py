import os
import numpy as np
from transformers import GPT2Tokenizer
from tqdm import tqdm

def tokenize_file(input_path, output_path, tokenizer):
    print(f"Processing {input_path}...")
    
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} not found.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Tokenize
    # We use the tokenizer to encode the text. 
    # GPT2Tokenizer does not add special tokens by default, but we might want to ensure <|endoftext|> is handled if present.
    # For simplicity and standard usage, we just encode.
    # Note: This loads the entire file into memory. For extremely large files, chunking would be better.
    # Given the file sizes (2.2GB for train), this might be tight on some systems but usually okay on servers.
    # Let's do a simple chunking to be safe and show progress.
    
    tokens = []
    chunk_size = 1024 * 1024  # 1MB chunks
    
    # Re-reading file in chunks to be memory safe
    with open(input_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            # We need to be careful with chunking not to split tokens. 
            # However, simple chunking is risky with BPE.
            # A safer approach for a simple script is to read lines if the data is line-based (like jsonl) or just read all if possible.
            # TinyStories is text. Let's try reading all for now as 2GB is usually manageable.
            pass
            
    # Let's stick to reading all for simplicity as it guarantees correctness of BPE merges across boundaries.
    print("Tokenizing (this may take a while)...")
    tokens = tokenizer.encode(text, verbose=False)
    
    # Convert to uint16
    # GPT-2 vocab size is 50257, which fits in uint16 (0-65535)
    tokens_np = np.array(tokens, dtype=np.uint16)
    
    print(f"Saving {len(tokens_np)} tokens to {output_path}...")
    fp = np.memmap(output_path, dtype=np.uint16, mode='w+', shape=tokens_np.shape)
    fp[:] = tokens_np[:]
    fp.flush()
    print("Done.")

def main():
    # Initialize tokenizer
    print("Loading GPT-2 tokenizer...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        print("Please ensure transformers is installed: pip install transformers")
        return

    # Define paths
    base_dir = 'data'
    files = [
        ('TinyStoriesV2-GPT4-train.txt', 'train.dat'),
        ('TinyStoriesV2-GPT4-valid.txt', 'valid.dat')
    ]

    for input_name, output_name in files:
        input_path = os.path.join(base_dir, input_name)
        output_path = os.path.join(base_dir, output_name)
        tokenize_file(input_path, output_path, tokenizer)

if __name__ == "__main__":
    main()
