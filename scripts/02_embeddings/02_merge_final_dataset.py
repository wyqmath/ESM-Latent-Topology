#!/usr/bin/env python3
# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
合并最终数据集：1706条现有序列 + 9362条ASTRAL 95序列 = 11068条

输入文件：
- data/all_sequences.fasta (1706条)
- data/all_metadata.csv
- data/astral95_supplement.fasta (8292条)
- data/astral95_supplement_metadata.csv

输出文件：
- data/all_sequences_final.fasta (11068条)
- data/metadata_final.csv
- data/final_dataset_statistics.png
- data/merge_report.txt

功能：合并序列、去冗余检查、统计分析、质量验证
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# 设置路径
data_dir = Path("data")

# 输入文件
existing_fasta = data_dir / "all_sequences.fasta"
existing_metadata = data_dir / "all_metadata.csv"
astral95_fasta = data_dir / "astral95_supplement.fasta"
astral95_metadata = data_dir / "astral95_supplement_metadata.csv"

# 输出文件
final_fasta = data_dir / "all_sequences_final.fasta"
final_metadata = data_dir / "metadata_final.csv"
final_stats_plot = data_dir / "final_dataset_statistics.png"
merge_report = data_dir / "merge_report.txt"

print("=" * 80)
print("合并最终数据集")
print("=" * 80)

# ============================================================================
# 1. 读取并合并FASTA文件
# ============================================================================
print("\n[1/5] 读取并合并FASTA文件...")

def read_fasta(fasta_path):
    """读取FASTA文件，返回序列字典"""
    sequences = {}
    current_id = None
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:]  # 去掉'>'
                current_seq = []
            else:
                current_seq.append(line)

        # 保存最后一条序列
        if current_id is not None:
            sequences[current_id] = ''.join(current_seq)

    return sequences

# 读取现有序列
existing_seqs = read_fasta(existing_fasta)
print(f"  现有序列数: {len(existing_seqs)}")

# 读取ASTRAL 95补充序列
astral95_seqs = read_fasta(astral95_fasta)
print(f"  ASTRAL 95补充序列数: {len(astral95_seqs)}")

# 合并序列
all_seqs = {**existing_seqs, **astral95_seqs}
print(f"  合并后总序列数: {len(all_seqs)}")

# 检查是否有重复的序列ID
if len(all_seqs) != len(existing_seqs) + len(astral95_seqs):
    print("  ⚠️  警告：检测到重复的序列ID！")
    existing_ids = set(existing_seqs.keys())
    astral95_ids = set(astral95_seqs.keys())
    duplicates = existing_ids & astral95_ids
    print(f"  重复ID数量: {len(duplicates)}")
    print(f"  示例: {list(duplicates)[:5]}")
else:
    print("  ✓ 无重复序列ID")

# 写入合并后的FASTA文件
print(f"\n  写入合并FASTA文件: {final_fasta}")
with open(final_fasta, 'w') as f:
    for seq_id, seq in all_seqs.items():
        f.write(f">{seq_id}\n")
        # 每行80个字符
        for i in range(0, len(seq), 80):
            f.write(seq[i:i+80] + '\n')

print(f"  ✓ 已写入 {len(all_seqs)} 条序列")

# ============================================================================
# 2. 读取并合并元数据
# ============================================================================
print("\n[2/5] 读取并合并元数据...")

# 读取现有元数据
df_existing = pd.read_csv(existing_metadata)
print(f"  现有元数据行数: {len(df_existing)}")

# 读取ASTRAL 95元数据
df_astral95 = pd.read_csv(astral95_metadata)
print(f"  ASTRAL 95元数据行数: {len(df_astral95)}")

# 合并元数据
df_final = pd.concat([df_existing, df_astral95], ignore_index=True)
print(f"  合并后元数据行数: {len(df_final)}")

# 检查是否有重复的序列ID
duplicate_ids = df_final[df_final.duplicated(subset=['seq_id'], keep=False)]
if len(duplicate_ids) > 0:
    print(f"  ⚠️  警告：元数据中检测到 {len(duplicate_ids)} 个重复ID")
else:
    print("  ✓ 元数据无重复ID")

# 写入合并后的元数据
print(f"\n  写入合并元数据文件: {final_metadata}")
df_final.to_csv(final_metadata, index=False)
print(f"  ✓ 已写入 {len(df_final)} 行元数据")

# ============================================================================
# 3. 统计分析
# ============================================================================
print("\n[3/5] 统计分析...")

# 类别统计
category_counts = df_final['category'].value_counts()
print("\n  类别分布:")
for cat, count in category_counts.items():
    print(f"    {cat}: {count} ({count/len(df_final)*100:.1f}%)")

# SCOP类别统计（仅针对astral95类别）
if 'scop_class' in df_final.columns:
    astral95_df = df_final[df_final['category'] == 'astral95']
    scop_counts = astral95_df['scop_class'].value_counts()
    print("\n  SCOP类别分布 (astral95):")
    for scop, count in scop_counts.items():
        print(f"    {scop}: {count} ({count/len(astral95_df)*100:.1f}%)")

# 长度统计
lengths = df_final['length'].values
print(f"\n  序列长度统计:")
print(f"    最小值: {lengths.min()}")
print(f"    最大值: {lengths.max()}")
print(f"    平均值: {lengths.mean():.1f} ± {lengths.std():.1f}")
print(f"    中位数: {np.median(lengths):.1f}")

# 按类别统计长度
print("\n  各类别长度统计:")
for cat in df_final['category'].unique():
    cat_lengths = df_final[df_final['category'] == cat]['length'].values
    print(f"    {cat}: {cat_lengths.mean():.1f} ± {cat_lengths.std():.1f} (n={len(cat_lengths)})")

# 氨基酸组成统计
print("\n  氨基酸组成分析...")
all_aa_counts = Counter()
total_residues = 0

for seq in all_seqs.values():
    all_aa_counts.update(seq)
    total_residues += len(seq)

aa_freq = {aa: count/total_residues*100 for aa, count in all_aa_counts.items()}
top_aa = sorted(aa_freq.items(), key=lambda x: x[1], reverse=True)[:10]

print("  前10种最常见氨基酸:")
for aa, freq in top_aa:
    print(f"    {aa}: {freq:.2f}%")

# ============================================================================
# 4. 存储预算检查
# ============================================================================
print("\n[4/5] 存储预算检查...")

n_sequences = len(all_seqs)
embedding_dim = 2560  # ESM-2 3B
bytes_per_value = 2  # FP16

# 序列级嵌入存储
seq_level_size_mb = (n_sequences * embedding_dim * bytes_per_value) / (1024**2)
print(f"  序列级嵌入 ({n_sequences}×{embedding_dim} FP16): {seq_level_size_mb:.2f} MB")

# 残基级嵌入存储（估算）
avg_length = lengths.mean()
residue_level_size_gb = (n_sequences * avg_length * embedding_dim * bytes_per_value) / (1024**3)
print(f"  残基级嵌入 ({n_sequences}×{avg_length:.0f}×{embedding_dim} FP16): {residue_level_size_gb:.2f} GB")

# 总存储估算
total_storage_gb = seq_level_size_mb / 1024 + residue_level_size_gb
print(f"\n  总存储估算: {total_storage_gb:.2f} GB")

if total_storage_gb < 60:
    print(f"  ✓ 存储预算充足 (预算: 60 GB, 使用: {total_storage_gb:.2f} GB)")
else:
    print(f"  ⚠️  警告：存储可能超出预算 (预算: 60 GB, 预计: {total_storage_gb:.2f} GB)")

# 推理时间估算
throughput = 12178  # 残基/秒 (来自benchmark)
total_residues = lengths.sum()
inference_time_sec = total_residues / throughput
inference_time_min = inference_time_sec / 60

print(f"\n  推理时间估算:")
print(f"    总残基数: {total_residues:,}")
print(f"    吞吐量: {throughput:,} 残基/秒")
print(f"    预计时间: {inference_time_min:.1f} 分钟 ({inference_time_sec:.0f} 秒)")

# ============================================================================
# 5. 生成统计图表
# ============================================================================
print("\n[5/5] 生成统计图表...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图1: 长度分布直方图
ax1 = axes[0, 0]
ax1.hist(lengths, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax1.axvline(lengths.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {lengths.mean():.1f}')
ax1.set_xlabel('Sequence Length (residues)', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title(f'Length Distribution (n={len(lengths)})', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# 子图2: 各类别长度箱线图
ax2 = axes[0, 1]
categories = df_final['category'].unique()
category_lengths = [df_final[df_final['category'] == cat]['length'].values for cat in categories]
bp = ax2.boxplot(category_lengths, labels=categories, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax2.set_ylabel('Sequence Length (residues)', fontsize=11)
ax2.set_title('Length Distribution by Category', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3, axis='y')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 子图3: 类别计数柱状图
ax3 = axes[1, 0]
category_counts_sorted = category_counts.sort_values(ascending=False)
bars = ax3.bar(range(len(category_counts_sorted)), category_counts_sorted.values, color='coral', alpha=0.7, edgecolor='black')
ax3.set_xticks(range(len(category_counts_sorted)))
ax3.set_xticklabels(category_counts_sorted.index, rotation=45, ha='right')
ax3.set_ylabel('Count', fontsize=11)
ax3.set_title('Sequence Count by Category', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3, axis='y')

# 在柱状图上添加数值标签
for i, (bar, count) in enumerate(zip(bars, category_counts_sorted.values)):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f'{count}\n({count/len(df_final)*100:.1f}%)',
             ha='center', va='bottom', fontsize=9)

# 子图4: 氨基酸组成柱状图
ax4 = axes[1, 1]
top_aa_sorted = sorted(top_aa, key=lambda x: x[1], reverse=True)
aa_names = [aa for aa, _ in top_aa_sorted]
aa_freqs = [freq for _, freq in top_aa_sorted]
ax4.bar(aa_names, aa_freqs, color='mediumseagreen', alpha=0.7, edgecolor='black')
ax4.set_xlabel('Amino Acid', fontsize=11)
ax4.set_ylabel('Frequency (%)', fontsize=11)
ax4.set_title('Top 10 Amino Acid Composition', fontsize=12, fontweight='bold')
ax4.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(final_stats_plot, dpi=300, bbox_inches='tight')
print(f"  ✓ 统计图表已保存: {final_stats_plot}")

# ============================================================================
# 6. 生成合并报告
# ============================================================================
print("\n[6/6] 生成合并报告...")

with open(merge_report, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("最终数据集合并报告\n")
    f.write("=" * 80 + "\n\n")

    f.write("## 数据来源\n")
    f.write(f"- 现有序列: {len(existing_seqs)} 条\n")
    f.write(f"- ASTRAL 95补充: {len(astral95_seqs)} 条\n")
    f.write(f"- 合并后总数: {len(all_seqs)} 条\n\n")

    f.write("## 类别分布\n")
    for cat, count in category_counts.items():
        f.write(f"- {cat}: {count} ({count/len(df_final)*100:.1f}%)\n")
    f.write("\n")

    if 'scop_class' in df_final.columns:
        f.write("## SCOP类别分布 (astral95)\n")
        astral95_df = df_final[df_final['category'] == 'astral95']
        scop_counts = astral95_df['scop_class'].value_counts()
        for scop, count in scop_counts.items():
            f.write(f"- {scop}: {count} ({count/len(astral95_df)*100:.1f}%)\n")
        f.write("\n")

    f.write("## 序列长度统计\n")
    f.write(f"- 最小值: {lengths.min()} 残基\n")
    f.write(f"- 最大值: {lengths.max()} 残基\n")
    f.write(f"- 平均值: {lengths.mean():.1f} ± {lengths.std():.1f} 残基\n")
    f.write(f"- 中位数: {np.median(lengths):.1f} 残基\n\n")

    f.write("## 各类别长度统计\n")
    for cat in df_final['category'].unique():
        cat_lengths = df_final[df_final['category'] == cat]['length'].values
        f.write(f"- {cat}: {cat_lengths.mean():.1f} ± {cat_lengths.std():.1f} (n={len(cat_lengths)})\n")
    f.write("\n")

    f.write("## 氨基酸组成 (前10)\n")
    for aa, freq in top_aa:
        f.write(f"- {aa}: {freq:.2f}%\n")
    f.write("\n")

    f.write("## 存储预算\n")
    f.write(f"- 序列级嵌入: {seq_level_size_mb:.2f} MB\n")
    f.write(f"- 残基级嵌入: {residue_level_size_gb:.2f} GB\n")
    f.write(f"- 总存储估算: {total_storage_gb:.2f} GB\n")
    f.write(f"- 预算状态: {'✓ 充足' if total_storage_gb < 60 else '⚠️ 可能超出'} (预算: 60 GB)\n\n")

    f.write("## 推理时间估算\n")
    f.write(f"- 总残基数: {total_residues:,}\n")
    f.write(f"- 吞吐量: {throughput:,} 残基/秒\n")
    f.write(f"- 预计时间: {inference_time_min:.1f} 分钟\n\n")

    f.write("## 质量检查\n")
    f.write(f"- 重复序列ID: {'无' if len(all_seqs) == len(existing_seqs) + len(astral95_seqs) else '有'}\n")
    f.write(f"- 元数据完整性: {'✓' if len(df_final) == len(all_seqs) else '⚠️'}\n")
    f.write(f"- 序列长度范围: {lengths.min()}-{lengths.max()} 残基\n")
    f.write(f"- 氨基酸种类: {len(all_aa_counts)} 种\n\n")

    f.write("=" * 80 + "\n")
    f.write("合并完成！\n")
    f.write("=" * 80 + "\n")

print(f"  ✓ 合并报告已保存: {merge_report}")

print("\n" + "=" * 80)
print("✓ 最终数据集合并完成！")
print("=" * 80)
print(f"\n输出文件:")
print(f"  - {final_fasta}")
print(f"  - {final_metadata}")
print(f"  - {final_stats_plot}")
print(f"  - {merge_report}")
print(f"\n数据集规模: {len(all_seqs)} 条序列")
print(f"存储预算: {total_storage_gb:.2f} GB / 60 GB")
print(f"推理时间: ~{inference_time_min:.1f} 分钟")
