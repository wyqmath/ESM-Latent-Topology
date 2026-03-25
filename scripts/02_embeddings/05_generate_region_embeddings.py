#!/usr/bin/env python3
# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
为 IDP/Fold-switching/Knotted 生成区域级 embeddings

策略：
1. 用全长序列生成 per-residue embeddings (保留上下文)
2. 根据区域标注切出核心区域的 embeddings
3. 对区域做平均池化得到区域级表征
4. 保存为新的 embedding 文件

修复记录 (2026-03-19):
  - Bug 1: 加 model.half() 与 07 脚本一致 (FP16 推理)
  - Bug 2: Fallback 改用 attention_mask pooling，排除 padding，包含 cls/eos (与 07 一致)
  - Bug 3: 截断后正确计算有效氨基酸长度，避免区域越界触发错误 fallback
  - Bug 4: BOS/EOS 一致性 — fallback 与 07 的 pooling 方式完全对齐

输入：
- data/all_sequences_final.fasta
- data/fold_switching_paper_regions.json
- data/idp_1000_regions.json
- data/knotted_286_regions.json

输出：
- data/embeddings/region_embeddings_extreme.pt (1365 × 2560)
- data/embeddings/region_embedding_index.csv
"""

import torch
import json
import pandas as pd
from pathlib import Path
from Bio import SeqIO
from tqdm import tqdm
from transformers import EsmModel, EsmTokenizer

# Config
DATA_DIR = Path("data")
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODEL_PATH = "/20232202212/.cache/huggingface/hub/models--facebook--esm2_t36_3B_UR50D/snapshots/476b639933c8baad5ad09a60ac1a87f987b656fc"

BATCH_SIZE = 4
MAX_LENGTH = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# Load model
print("Loading ESM-2 3B model...")
model = EsmModel.from_pretrained(MODEL_PATH).to(DEVICE)
# FIX 1: match 07 script — FP16 inference
if DEVICE == "cuda":
    model = model.half()
    print("Model converted to FP16 (matching 07_batch_inference_final)")
tokenizer = EsmTokenizer.from_pretrained(MODEL_PATH)
model.eval()

# Load region annotations
print("\nLoading region annotations...")
with open(DATA_DIR / "fold_switching_paper_regions.json") as f:
    fs_data = json.load(f)
with open(DATA_DIR / "idp_1000_regions.json") as f:
    idp_data = json.load(f)
with open(DATA_DIR / "knotted_286_regions.json") as f:
    knotted_data = json.load(f)

# Merge all regions
all_regions = {}
for item in fs_data['sequences']:
    all_regions[item['id']] = item['regions']
for item in idp_data['sequences']:
    all_regions[item['id']] = item['regions']
for item in knotted_data['sequences']:
    all_regions[item['id']] = item['regions']

print(f"Total sequences with regions: {len(all_regions)}")

# Load sequences
print("\nLoading sequences...")
seq_dict = {}
for record in SeqIO.parse(DATA_DIR / "all_sequences_final.fasta", "fasta"):
    parts = record.id.split('|')
    if len(parts) >= 2:
        base_id = parts[1]
        if base_id in all_regions:
            seq_dict[record.id] = (str(record.seq), all_regions[base_id])

print(f"Loaded {len(seq_dict)} sequences")

# Process in batches
region_embeddings = []
seq_ids_ordered = []

# Stats tracking
stats = {'total': 0, 'region_ok': 0, 'partial_truncated': 0, 'fallback': 0}

print("\nGenerating region embeddings...")
seq_items = list(seq_dict.items())

for i in tqdm(range(0, len(seq_items), BATCH_SIZE)):
    batch_items = seq_items[i:i+BATCH_SIZE]
    batch_seqs = [seq for _, (seq, _) in batch_items]
    batch_ids = [sid for sid, _ in batch_items]

    inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True,
                      truncation=True, max_length=MAX_LENGTH).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, 2560)

    attention_mask = inputs['attention_mask']  # (batch, seq_len)

    for j, seq_id in enumerate(batch_ids):
        seq_str, regions = seq_dict[seq_id]
        stats['total'] += 1

        # FIX 3: compute actual amino acid length after truncation
        # ESM-2 tokenizes as: [CLS] aa1 aa2 ... aaN [EOS] [PAD...]
        # max_length=1024 means at most 1022 amino acids + CLS + EOS
        actual_aa_len = min(len(seq_str), MAX_LENGTH - 2)

        # Collect region token indices (in hidden_states space, +1 for CLS)
        region_tokens = []
        for region in regions:
            start = region['start'] - 1  # convert 1-indexed to 0-indexed
            end = region['end']          # end is inclusive in annotation, exclusive here

            # FIX 3: skip region parts that fall beyond truncation boundary
            if start >= actual_aa_len:
                continue
            valid_end = min(end, actual_aa_len)

            # +1 offset for CLS token at position 0
            region_tokens.extend(range(start + 1, valid_end + 1))

        # Deduplicate and sort
        region_tokens = sorted(set(region_tokens))

        if region_tokens:
            region_emb = hidden_states[j, region_tokens, :].mean(dim=0)
            if len(seq_str) > actual_aa_len:
                stats['partial_truncated'] += 1
            else:
                stats['region_ok'] += 1
        else:
            # FIX 2 & 4: fallback uses attention_mask pooling (same as 07 script)
            # This includes CLS and EOS, excludes PAD — exactly matching 07's behavior
            mask_j = attention_mask[j].unsqueeze(-1)  # (seq_len, 1)
            masked_hidden = hidden_states[j] * mask_j  # (seq_len, 2560)
            region_emb = masked_hidden.sum(dim=0) / mask_j.sum(dim=0)
            stats['fallback'] += 1

        region_embeddings.append(region_emb.cpu().half())
        seq_ids_ordered.append(seq_id)

# Save
region_embeddings = torch.stack(region_embeddings)
print(f"\nRegion embeddings shape: {region_embeddings.shape}")

EMBEDDINGS_DIR.mkdir(exist_ok=True)
torch.save(region_embeddings, EMBEDDINGS_DIR / "region_embeddings_extreme.pt")

index_df = pd.DataFrame({'seq_id': seq_ids_ordered, 'index': range(len(seq_ids_ordered))})
index_df.to_csv(EMBEDDINGS_DIR / "region_embedding_index.csv", index=False)

print(f"\nSaved to {EMBEDDINGS_DIR / 'region_embeddings_extreme.pt'}")
print(f"Saved index to {EMBEDDINGS_DIR / 'region_embedding_index.csv'}")

print(f"\n{'='*60}")
print(f"Statistics:")
print(f"  Total processed:     {stats['total']}")
print(f"  Region OK:           {stats['region_ok']}")
print(f"  Partial truncated:   {stats['partial_truncated']} (region partially within truncation window)")
print(f"  Fallback (attn_mask):{stats['fallback']} (all regions beyond truncation → full-seq pooling)")
print(f"{'='*60}")
