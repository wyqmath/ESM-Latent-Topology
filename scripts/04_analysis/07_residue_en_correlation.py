# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
残基级嵌入与 E[n] 相关性分析（全量版）

覆盖 anchor + astral95 + integrable 三类有 PDB 结构的序列 (约 9,195 条)。
对每条序列：
  1. 从 PDB 提取 Cα 坐标 → Frenet 标架 → 残基级 E[n]
  2. 从 ESM-2 提取残基级嵌入 → 相邻残基余弦距离
  3. 计算两者的 Spearman 相关性

输入:
- data/metadata_final_with_en.csv
- data/embeddings/embedding_index_final.csv
- data/all_sequences_final.fasta
- data/pdb_files/*.pdb

输出:
- data/residue_en_correlation_full.csv
"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from pathlib import Path
from Bio import SeqIO

# ============ 配置 ============
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "/20232202212/.cache/huggingface/hub/models--facebook--esm2_t36_3B_UR50D/snapshots/476b639933c8baad5ad09a60ac1a87f987b656fc"
PDB_DIR = Path('data/pdb_files')
MAX_SEQ_LEN = 1022


# ============ 数学函数 ============
def compute_frenet_frame(coords):
    """Frenet 标架 → 曲率 kappa、挠率 tau"""
    N = len(coords)
    if N < 4:
        return None, None
    r_prime = np.diff(coords, axis=0)
    r_double_prime = np.diff(r_prime, axis=0)
    T = r_prime[:-1]
    T_norm = np.linalg.norm(T, axis=1, keepdims=True)
    T = T / (T_norm + 1e-10)
    kappa_N = r_double_prime
    kappa = np.linalg.norm(kappa_N, axis=1)
    N_vec = kappa_N / (kappa[:, None] + 1e-10)
    B = np.cross(T, N_vec)
    dB_ds = np.diff(B, axis=0)
    tau = -np.sum(dB_ds * N_vec[:-1], axis=1)
    tau_aligned = np.concatenate([[tau[0]], tau])
    return kappa, tau_aligned


def compute_residue_en(coords):
    """残基级 E[n] 可积性误差"""
    kappa, tau = compute_frenet_frame(coords)
    if kappa is None or len(kappa) < 2:
        return None
    theta_kappa = np.arctan2(np.diff(kappa), kappa[:-1] + 1e-10)
    theta_tau = np.arctan2(np.diff(tau), tau[:-1] + 1e-10)
    cos_product = np.clip(np.cos(theta_kappa) * np.cos(theta_tau), -1.0, 1.0)
    return np.arccos(cos_product)


def load_pdb_ca_coords(pdb_file):
    """从 PDB 文件加载 Cα 坐标"""
    coords = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords) if coords else np.array([]).reshape(0, 3)


# ============ 主流程 ============
print("=" * 60)
print("残基级嵌入与 E[n] 相关性分析 (全量版)")
print("=" * 60)

# Step 1: 匹配有 PDB 文件的序列
print("\n[1/4] Identifying sequences with PDB files...")
metadata = pd.read_csv('data/metadata_final_with_en.csv')
embed_index = pd.read_csv('data/embeddings/embedding_index_final.csv')
pdb_files_available = set(f.replace('.pdb', '').lower() for f in _os.listdir(PDB_DIR))

seq_pdb_map = {}
for _, row in metadata.iterrows():
    sid = row['seq_id']
    parts = sid.split('|')
    cat = row['category']
    if cat == 'anchor':
        if len(parts) >= 3:
            pdb4 = parts[2][:4].lower()
            if pdb4 in pdb_files_available:
                seq_pdb_map[sid] = PDB_DIR / f"{pdb4}.pdb"
    elif cat == 'astral95':
        if len(parts) >= 3:
            scop_id = parts[2]
            if len(scop_id) >= 5:
                pdb4 = scop_id[1:5].lower()
                if pdb4 in pdb_files_available:
                    seq_pdb_map[sid] = PDB_DIR / f"{pdb4}.pdb"
    elif cat == 'integrable':
        if len(parts) >= 3:
            pdb_id = parts[2]
            pdb4 = pdb_id[:4].lower() if len(pdb_id) >= 4 else pdb_id.lower()
            if pdb4 in pdb_files_available:
                seq_pdb_map[sid] = PDB_DIR / f"{pdb4}.pdb"
print(f"  Sequences with PDB files: {len(seq_pdb_map)}")

# Step 2: 加载序列
print("\n[2/4] Loading sequences from FASTA...")
seq_dict = {}
for record in SeqIO.parse('data/all_sequences_final.fasta', 'fasta'):
    seq_dict[record.id] = str(record.seq)
valid_seqs = [(s, p, seq_dict[s]) for s, p in seq_pdb_map.items() if s in seq_dict]
print(f"  Valid (PDB + FASTA): {len(valid_seqs)}")

# Step 3: ESM-2 残基级嵌入 + E[n] 相关性
print(f"\n[3/4] Loading ESM-2 on {DEVICE}...")
from transformers import EsmModel, EsmTokenizer
tokenizer = EsmTokenizer.from_pretrained(MODEL_PATH)
model = EsmModel.from_pretrained(MODEL_PATH)
model = model.to(DEVICE)
model.eval()
print("  Model ready")

results = []
skipped = {'no_en': 0, 'short': 0, 'mismatch': 0, 'error': 0, 'truncated': 0}

for i, (sid, pdb_path, sequence) in enumerate(valid_seqs):
    try:
        ca_coords = load_pdb_ca_coords(pdb_path)
        if len(ca_coords) < 5:
            skipped['short'] += 1
            continue

        en_residue = compute_residue_en(ca_coords)
        if en_residue is None or len(en_residue) < 5:
            skipped['no_en'] += 1
            continue

        seq_for_model = sequence[:MAX_SEQ_LEN]
        if len(sequence) > MAX_SEQ_LEN:
            skipped['truncated'] += 1

        inputs = tokenizer(seq_for_model, return_tensors="pt", padding=False,
                           truncation=True, max_length=MAX_SEQ_LEN + 2)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            residue_embed = outputs.hidden_states[-1][0, 1:-1, :].cpu().numpy()

        L_embed = len(residue_embed)
        if L_embed < 5:
            skipped['short'] += 1
            continue

        cosine_distances = np.array([cosine(residue_embed[j], residue_embed[j + 1])
                                     for j in range(L_embed - 1)])

        min_len = min(len(cosine_distances), len(en_residue))
        if min_len < 5:
            skipped['mismatch'] += 1
            continue

        cos_aligned = cosine_distances[:min_len]
        en_aligned = en_residue[:min_len]
        rho, pval = spearmanr(cos_aligned, en_aligned)

        results.append({
            'seq_id': sid,
            'spearman_rho': rho,
            'p_value': pval,
            'mean_E_n': np.mean(en_aligned),
            'length': len(sequence),
            'n_residues_aligned': min_len,
            'category': metadata[metadata['seq_id'] == sid]['category'].iloc[0],
        })

        if (i + 1) % 200 == 0:
            print(f"  [{i + 1}/{len(valid_seqs)}] {len(results)} successful")

    except Exception as e:
        skipped['error'] += 1
        if skipped['error'] <= 5:
            print(f"  Error on {sid}: {e}")

# Step 4: 保存结果
print("\n[4/4] Saving results...")
results_df = pd.DataFrame(results)
results_df.to_csv('data/residue_en_correlation_full.csv', index=False)

print(f"\nDONE: {len(results_df)} sequences analyzed out of {len(valid_seqs)}")
print(f"Skipped: short={skipped['short']}, no_en={skipped['no_en']}, "
      f"mismatch={skipped['mismatch']}, error={skipped['error']}, "
      f"truncated={skipped['truncated']}")
if len(results_df) > 0:
    print(f"Mean rho: {results_df['spearman_rho'].mean():.4f} "
          f"+/- {results_df['spearman_rho'].std():.4f}")
    print(f"Median rho: {results_df['spearman_rho'].median():.4f}")
    sig = (results_df['p_value'] < 0.05).sum()
    print(f"Significant (p<0.05): {sig} ({sig / len(results_df) * 100:.1f}%)")
    for cat in sorted(results_df['category'].unique()):
        sub = results_df[results_df['category'] == cat]
        print(f"  {cat}: n={len(sub)}, mean rho={sub['spearman_rho'].mean():.4f}")
