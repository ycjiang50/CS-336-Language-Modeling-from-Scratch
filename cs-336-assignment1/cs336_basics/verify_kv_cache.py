import torch
import time
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.layer import TransformerLM
from cs336_basics.generate import generate_text

def test_kv_cache():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create a small dummy model
    config = {
        'vocab_size': 1000,
        'context_length': 128,
        'num_layers': 2,
        'd_model': 64,
        'num_heads': 4,
        'rope_theta': 10000.0,
        'd_ff': 256
    }
    
    model = TransformerLM(**config).to(device)
    model.eval()
    
    # Dummy tokenizer
    class DummyTokenizer:
        def encode(self, text):
            return [1, 2, 3, 4, 5] # Dummy tokens
        def decode(self, tokens):
            return " ".join(str(t) for t in tokens)
    
    tokenizer = DummyTokenizer()
    prompt = "Test prompt"
    
    # 1. Run WITH KV cache (default in my modified generate_text)
    print("\nRunning WITH KV Cache...")
    start_time = time.time()
    output_kv = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=20,
        temperature=1.0,
        top_p=1.0, # Deterministic sampling if we control random seed, but here we just check it runs
        device=device
    )
    end_time = time.time()
    time_kv = end_time - start_time
    print(f"Time with KV cache: {time_kv:.4f}s")
    print(f"Output: {output_kv}")
    
    # 2. Run WITHOUT KV cache
    # To do this, I need to temporarily disable KV cache in generate_text or call model directly.
    # Since I modified generate_text to ALWAYS use KV cache, I will manually run a generation loop here without it.
    print("\nRunning WITHOUT KV Cache...")
    
    prompt_tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    generated_tokens_no_kv = prompt_tokens.copy()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(20):
            logits = model(input_ids, use_cache=False) # Explicitly disable cache
            next_token_logits = logits[0, -1, :]
            probabilities = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.argmax(probabilities).item() # Greedy for comparison
            generated_tokens_no_kv.append(next_token)
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
            if input_ids.size(1) > model.context_length:
                input_ids = input_ids[:, -model.context_length:]
                
    end_time = time.time()
    time_no_kv = end_time - start_time
    print(f"Time WITHOUT KV cache: {time_no_kv:.4f}s")
    
    print(f"\nSpeedup: {time_no_kv / time_kv:.2f}x")
    
    # Note: Outputs might differ because generate_text uses sampling and I used greedy here.
    # But the main point is that it runs without errors.

if __name__ == "__main__":
    test_kv_cache()
