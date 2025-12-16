from cs336_basics.bpe import *
import yaml


def save_tokenizer_yaml(vocab, merges, fname):
    "Save vocab and merges to a YAML file with UTF-8 decoding for readability."
    # Convert bytes → string for readability
    vocab_serializable = {
        k: v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v
        for k, v in vocab.items()
    }
    merges_serializable = [
        (a.decode("utf-8", errors="replace"), b.decode("utf-8", errors="replace"))
        for a, b in merges
    ]
    
    with open(fname, "w", encoding="utf-8") as f:
        yaml.dump(
            {"vocab": vocab_serializable, "merges": merges_serializable},
            f,
            allow_unicode=True,
            sort_keys=False
        )

def load_tokenizer_yaml(fname):
    "Load vocab and merges from a YAML file, converting strings back to bytes."
    with open(fname, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    vocab_loaded = {
        int(k): v.encode("utf-8") if isinstance(v, str) else v
        for k, v in data["vocab"].items()
    }
    merges_loaded = [
        (a.encode("utf-8"), b.encode("utf-8")) for a, b in data["merges"]
    ]
    return vocab_loaded, merges_loaded

vocab, merges = train_bpe(
        input_path='data/TinyStoriesV2-GPT4-valid.txt',
        vocab_size=10_000,
        special_tokens=["<|endoftext|>","<|endoftext|><|endoftext|>"],
    )

save_tokenizer_yaml(vocab,merges,'tokenizer_owt_valid.yaml')