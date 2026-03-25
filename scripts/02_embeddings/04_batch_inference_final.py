#!/usr/bin/env python3
# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
ESM-2 3B 批量推理脚本 (最终数据集 11068条序列)

输入文件：
  - data/all_sequences_final.fasta (11068条序列)

输出文件：
  - data/embeddings/sequence_embeddings_final.pt (11068×2560 FP16张量)
  - data/embeddings/embedding_index_final.csv (序列ID到索引的映射)
  - data/embeddings/progress_final.json (断点续传进度)

功能描述：
  - 使用ESM-2 3B模型提取序列级嵌入（平均池化）
  - 支持断点续传，可从中断处继续
  - 使用最优配置：batch_size=8
  - FP16格式节省存储和显存
  - 实时显示进度和预估完成时间
"""

import torch
import json
import time
import pandas as pd
from pathlib import Path
from Bio import SeqIO
from tqdm import tqdm
from transformers import EsmModel, EsmTokenizer

# ============================================================================
# 配置参数
# ============================================================================

# 路径配置
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
FASTA_FILE = DATA_DIR / "all_sequences_final.fasta"
MODEL_PATH = "/20232202212/.cache/huggingface/hub/models--facebook--esm2_t36_3B_UR50D/snapshots/476b639933c8baad5ad09a60ac1a87f987b656fc"

# 输出文件
OUTPUT_EMBEDDINGS = EMBEDDINGS_DIR / "sequence_embeddings_final.pt"
OUTPUT_INDEX = EMBEDDINGS_DIR / "embedding_index_final.csv"
PROGRESS_FILE = EMBEDDINGS_DIR / "progress_final.json"

# 推理配置
BATCH_SIZE = 8  # 基准测试确定的最优值
MAX_LENGTH = 1024  # ESM-2最大序列长度
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# 辅助函数
# ============================================================================

def load_sequences(fasta_file):
    """加载FASTA文件中的所有序列"""
    sequences = []
    seq_ids = []

    print(f"Loading sequences from {fasta_file}...")
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq_ids.append(record.id)
        sequences.append(str(record.seq))

    print(f"Loaded {len(sequences)} sequences")
    return seq_ids, sequences


def load_progress():
    """加载断点续传进度"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            progress = json.load(f)
        print(f"Resuming from sequence {progress['last_completed'] + 1}/{progress['total_sequences']}")
        return progress
    return None


def save_progress(last_completed, total_sequences):
    """保存断点续传进度"""
    progress = {
        'last_completed': last_completed,
        'total_sequences': total_sequences,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def extract_embeddings_batch(model, tokenizer, sequences, device):
    """
    批量提取序列级嵌入

    Args:
        model: ESM-2模型
        tokenizer: ESM-2分词器
        sequences: 序列列表
        device: 计算设备

    Returns:
        embeddings: (batch_size, 2560) 张量
    """
    # 分词（自动截断到MAX_LENGTH）
    inputs = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    ).to(device)

    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs)
        # 获取最后一层隐藏状态: (batch_size, seq_len, 2560)
        hidden_states = outputs.last_hidden_state

        # 创建attention mask（排除padding token）
        attention_mask = inputs['attention_mask'].unsqueeze(-1)  # (batch_size, seq_len, 1)

        # 平均池化（只对非padding位置）
        masked_hidden = hidden_states * attention_mask
        sum_hidden = masked_hidden.sum(dim=1)  # (batch_size, 2560)
        sum_mask = attention_mask.sum(dim=1)  # (batch_size, 1)
        embeddings = sum_hidden / sum_mask  # (batch_size, 2560)

    return embeddings.cpu().half()  # 转换为FP16并移到CPU


# ============================================================================
# 主函数
# ============================================================================

def main():
    # 创建输出目录
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    # 加载序列
    seq_ids, sequences = load_sequences(FASTA_FILE)
    total_sequences = len(sequences)

    # 检查断点续传
    progress = load_progress()
    if progress is not None:
        start_idx = progress['last_completed'] + 1
        # 加载已有的嵌入
        if OUTPUT_EMBEDDINGS.exists():
            existing_embeddings = torch.load(OUTPUT_EMBEDDINGS)
            print(f"Loaded existing embeddings: {existing_embeddings.shape}")
        else:
            print("Warning: Progress file exists but embeddings file not found. Starting from scratch.")
            start_idx = 0
            existing_embeddings = None
    else:
        start_idx = 0
        existing_embeddings = None

    # 如果已经全部完成
    if start_idx >= total_sequences:
        print("All sequences already processed!")
        return

    # 加载模型
    print(f"\nLoading ESM-2 3B model from {MODEL_PATH}...")
    print(f"Device: {DEVICE}")

    tokenizer = EsmTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = EsmModel.from_pretrained(MODEL_PATH, local_files_only=True)
    model = model.to(DEVICE)
    model.eval()

    # 转换为FP16以节省显存
    if DEVICE == "cuda":
        model = model.half()
        print("Model converted to FP16")

    print(f"Model loaded successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # 准备存储所有嵌入
    all_embeddings = []
    if existing_embeddings is not None:
        all_embeddings.append(existing_embeddings)

    # 批量推理
    print(f"\nStarting batch inference...")
    print(f"Total sequences: {total_sequences}")
    print(f"Starting from: {start_idx}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Estimated time: {(total_sequences - start_idx) / BATCH_SIZE * 0.5 / 60:.1f} minutes")
    print("-" * 80)

    start_time = time.time()

    # 使用tqdm显示进度
    with tqdm(total=total_sequences - start_idx, desc="Processing", unit="seq") as pbar:
        for i in range(start_idx, total_sequences, BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, total_sequences)
            batch_sequences = sequences[i:batch_end]

            # 提取嵌入
            batch_embeddings = extract_embeddings_batch(
                model, tokenizer, batch_sequences, DEVICE
            )
            all_embeddings.append(batch_embeddings)

            # 更新进度
            pbar.update(batch_end - i)

            # 每10个batch保存一次进度
            if (i // BATCH_SIZE) % 10 == 0:
                # 合并所有嵌入并保存
                embeddings_tensor = torch.cat(all_embeddings, dim=0)
                torch.save(embeddings_tensor, OUTPUT_EMBEDDINGS)
                save_progress(batch_end - 1, total_sequences)

    # 最终保存
    print("\nSaving final results...")
    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    torch.save(embeddings_tensor, OUTPUT_EMBEDDINGS)

    # 保存索引文件
    index_df = pd.DataFrame({
        'seq_id': seq_ids,
        'index': range(len(seq_ids))
    })
    index_df.to_csv(OUTPUT_INDEX, index=False)

    # 删除进度文件（已完成）
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()

    # 统计信息
    elapsed_time = time.time() - start_time
    total_residues = sum(len(seq) for seq in sequences)
    throughput = total_residues / elapsed_time

    print("\n" + "=" * 80)
    print("Batch inference completed!")
    print("=" * 80)
    print(f"Total sequences: {total_sequences}")
    print(f"Embedding shape: {embeddings_tensor.shape}")
    print(f"Embedding dtype: {embeddings_tensor.dtype}")
    print(f"File size: {OUTPUT_EMBEDDINGS.stat().st_size / 1024**2:.2f} MB")
    print(f"Elapsed time: {elapsed_time / 60:.2f} minutes")
    print(f"Throughput: {throughput:.0f} residues/second")
    print(f"Average time per sequence: {elapsed_time / total_sequences:.3f} seconds")
    print(f"\nOutput files:")
    print(f"  - {OUTPUT_EMBEDDINGS}")
    print(f"  - {OUTPUT_INDEX}")
    print("=" * 80)


if __name__ == "__main__":
    main()
