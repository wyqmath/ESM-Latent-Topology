#!/usr/bin/env python3
# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
合并区域 embeddings 到最终数据集

策略：
1. 加载原始 sequence_embeddings_final.pt (11068 × 2560)
2. 备份全长版本为 sequence_embeddings_fulllength.pt
3. 加载 region_embeddings_extreme.pt (1365 × 2560)
4. 根据 seq_id 匹配，替换 extreme 类别的 embeddings
5. 保存替换后版本为 sequence_embeddings_region_replaced.pt
6. 同时覆盖 sequence_embeddings_final.pt（向后兼容）
7. 更新 metadata 标记哪些使用了区域 embeddings

输出：
  - data/embeddings/sequence_embeddings_fulllength.pt  (全长版本备份)
  - data/embeddings/sequence_embeddings_region_replaced.pt (区域替换版本)
  - data/embeddings/sequence_embeddings_final.pt (= region_replaced，向后兼容)
"""

import torch
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Load original embeddings
print("Loading original embeddings...")
orig_emb = torch.load(EMBEDDINGS_DIR / "sequence_embeddings_final.pt")
orig_index = pd.read_csv(EMBEDDINGS_DIR / "embedding_index_final.csv")
print(f"Original shape: {orig_emb.shape}")

# Save full-length backup BEFORE replacement
fulllength_path = EMBEDDINGS_DIR / "sequence_embeddings_fulllength.pt"
torch.save(orig_emb.clone(), fulllength_path)
print(f"Saved full-length backup: {fulllength_path}")

# Load region embeddings
print("\nLoading region embeddings...")
region_emb = torch.load(EMBEDDINGS_DIR / "region_embeddings_extreme.pt")
region_index = pd.read_csv(EMBEDDINGS_DIR / "region_embedding_index.csv")
print(f"Region shape: {region_emb.shape}")

# Create mapping
region_map = dict(zip(region_index['seq_id'], range(len(region_index))))

# Replace embeddings
replaced_count = 0
for i, seq_id in enumerate(orig_index['seq_id']):
    if seq_id in region_map:
        region_idx = region_map[seq_id]
        orig_emb[i] = region_emb[region_idx]
        replaced_count += 1

print(f"\nReplaced {replaced_count} embeddings")

# Save region-replaced version
region_replaced_path = EMBEDDINGS_DIR / "sequence_embeddings_region_replaced.pt"
torch.save(orig_emb, region_replaced_path)
print(f"Saved region-replaced: {region_replaced_path}")

# Also save as final (backward compatibility)
torch.save(orig_emb, EMBEDDINGS_DIR / "sequence_embeddings_final.pt")
print(f"Saved to {EMBEDDINGS_DIR / 'sequence_embeddings_final.pt'}")

# Update metadata
metadata = pd.read_csv(DATA_DIR / "metadata_final_with_en.csv")
metadata['uses_region_embedding'] = metadata['seq_id'].isin(region_index['seq_id'])
metadata.to_csv(DATA_DIR / "metadata_final_with_en.csv", index=False)
print(f"Updated metadata with region embedding flag")

print(f"\n{'='*60}")
print(f"Summary:")
print(f"  Full-length backup:  {fulllength_path}")
print(f"  Region-replaced:     {region_replaced_path}")
print(f"  Final (=replaced):   {EMBEDDINGS_DIR / 'sequence_embeddings_final.pt'}")
print(f"  Replaced {replaced_count} extreme embeddings with region embeddings")
print(f"{'='*60}")
