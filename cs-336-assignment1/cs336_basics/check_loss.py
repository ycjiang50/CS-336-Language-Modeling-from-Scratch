import torch
import os
import sys
import numpy as np
from cs336_basics.layer import *
from cs336_basics.trainer.utils import cross_entropy
from cs336_basics.trainer.data_loading import data_loading

config = {
    'vocab_size': 50257,
    'context_len': 256,
    'num_layers': 4,
    'd_model': 256,
    'num_heads': 16,
    'rope_theta': 10000.0,
    'd_ff': 1024,
    'batch_size': 32
}

def get_dataset_memmap(path, dtype=np.uint16):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return np.memmap(path, dtype=dtype, mode='r')

def main():
    device = 'cpu' # 既然你在用 CPU 训练，这里也用 CPU
    
    # 1. 检查 Checkpoint 是否存在
    checkpoint_path = './checkpoints/checkpoint_5000.pt' # 等到这个文件生成了再跑！
    if not os.path.exists(checkpoint_path):
        print(f"Waiting... {checkpoint_path} not found yet.")
        return

    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # 2. 初始化模型
    model = TransformerLM(
        vocab_size=config['vocab_size'],
        context_length=config['context_len'],
        num_layers=config['num_layers'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        rope_theta=config['rope_theta'],
        d_ff=config['d_ff']
    ).to(device)

    # 3. 加载权重
    # 注意：save_checkpoint 存的是字典 {'model': ..., 'optimizer': ...}
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # 4. 加载验证数据
    val_data_path = './data/valid.dat' # 确保路径对
    val_data = get_dataset_memmap(val_data_path)
    
    print("Running validation...")
    val_losses = []
    
    # 跑 10 个 batch 看看效果
    with torch.no_grad():
        for i in range(10):
            val_input_ids, val_target_ids = data_loading(
                val_data,
                config['batch_size'],
                config['context_len'],
                device=device
            )
            
            # 转 Long 类型
            val_input_ids = val_input_ids.long()
            val_target_ids = val_target_ids.long()
            
            logits = model(val_input_ids)
            
            # Flatten
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = val_target_ids.view(-1)
            
            loss = cross_entropy(logits_flat, targets_flat)
            val_losses.append(loss.item())
            print(f"Batch {i+1}/10 Loss: {loss.item():.4f}")

    avg_loss = np.mean(val_losses)
    print(f"\nFinal Validation Loss: {avg_loss:.4f}")
    print(f"Perplexity: {np.exp(avg_loss):.2f}")

if __name__ == '__main__':
    main()