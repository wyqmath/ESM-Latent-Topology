#!/usr/bin/env python3
# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
扩展数据集并进行细粒度分析

输入文件:
- astral-scopedom-seqres-gd-sel-gs-bib-95-2.08.fa (35,494条ASTRAL 95序列)
- data/all_sequences.fasta (1,706条现有序列)
- data/metadata_complete.csv (现有元数据)

输出文件:
- data/astral95_analysis.txt (95数据集分析报告)
- data/astral95_supplement.fasta (筛选后的补充序列)
- data/astral95_supplement_metadata.csv (补充序列元数据)
- data/expansion_report.txt (扩展统计报告)

功能描述:
1. 分析ASTRAL 95数据集的规模、长度分布、SCOP分类
2. 从95数据集中大规模筛选补充序列（目标：扩展到10,000+条）
3. 筛选标准：长度40-500残基，氨基酸组成正常，与现有序列去冗余
4. 细粒度分析：总数据量统计、子数据集差异、按SCOP类别/长度/来源划分
"""

import re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO

# 路径配置
# 路径配置 (chdir 已切换到项目根目录)
ASTRAL95_FILE = Path("data/raw/astral-scopedom-seqres-gd-sel-gs-bib-95-2.08.fa")
EXISTING_FASTA = Path("data/all_sequences.fasta")
EXISTING_METADATA = Path("data/metadata_complete.csv")
OUTPUT_DIR = Path("data")

# 输出文件
ANALYSIS_REPORT = OUTPUT_DIR / "astral95_analysis.txt"
SUPPLEMENT_FASTA = OUTPUT_DIR / "astral95_supplement.fasta"
SUPPLEMENT_METADATA = OUTPUT_DIR / "astral95_supplement_metadata.csv"
EXPANSION_REPORT = OUTPUT_DIR / "expansion_report.txt"

# 筛选参数
MIN_LENGTH = 40
MAX_LENGTH = 500
TARGET_TOTAL = 10000  # 目标总序列数（现有1706 + 补充~8300）
KMER_SIZE = 5
SIMILARITY_THRESHOLD = 0.95  # 95%相似度阈值（比40数据集更宽松）

# 标准氨基酸
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

def parse_scop_class(header):
    """从ASTRAL header解析SCOP分类"""
    # 格式: >d1ux8a_ a.1.1.1 (A:) Description
    match = re.search(r'([a-g])\.\d+\.\d+\.\d+', header)
    if match:
        class_code = match.group(1)
        class_map = {
            'a': 'all-alpha',
            'b': 'all-beta',
            'c': 'alpha/beta',
            'd': 'alpha+beta',
            'e': 'multi-domain',
            'f': 'membrane',
            'g': 'small'
        }
        return class_map.get(class_code, 'unknown')
    return 'unknown'

def kmer_profile(seq, k=5):
    """计算k-mer频率向量"""
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    counts = Counter(kmers)
    total = len(kmers)
    return {kmer: count/total for kmer, count in counts.items()}

def kmer_similarity(profile1, profile2):
    """计算两个k-mer profile的相似度（Jaccard系数）"""
    all_kmers = set(profile1.keys()) | set(profile2.keys())
    if not all_kmers:
        return 0.0
    intersection = sum(min(profile1.get(k, 0), profile2.get(k, 0)) for k in all_kmers)
    union = sum(max(profile1.get(k, 0), profile2.get(k, 0)) for k in all_kmers)
    return intersection / union if union > 0 else 0.0

def check_aa_composition(seq):
    """检查氨基酸组成是否正常"""
    aa_counts = Counter(seq)
    # 检查是否只包含标准氨基酸
    if not set(seq).issubset(STANDARD_AA):
        return False
    # 检查是否有单个氨基酸占比过高（>50%）
    for aa, count in aa_counts.items():
        if count / len(seq) > 0.5:
            return False
    return True

print("=" * 80)
print("ASTRAL 95数据集扩展与细粒度分析")
print("=" * 80)

# ============================================================================
# 步骤1: 分析ASTRAL 95数据集
# ============================================================================
print("\n[步骤1] 分析ASTRAL 95数据集...")

astral95_records = []
scop_classes = []
lengths = []
aa_composition = Counter()

for record in SeqIO.parse(ASTRAL95_FILE, "fasta"):
    seq = str(record.seq).upper()
    scop_class = parse_scop_class(record.description)

    astral95_records.append({
        'id': record.id,
        'description': record.description,
        'seq': seq,
        'length': len(seq),
        'scop_class': scop_class
    })

    scop_classes.append(scop_class)
    lengths.append(len(seq))
    aa_composition.update(seq)

print(f"✓ 总序列数: {len(astral95_records)}")
print(f"✓ 长度范围: {min(lengths)}-{max(lengths)} 残基")
print(f"✓ 平均长度: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")

# SCOP分类统计
scop_counts = Counter(scop_classes)
print(f"\n✓ SCOP分类分布:")
for scop_class, count in scop_counts.most_common():
    print(f"  - {scop_class}: {count} ({count/len(astral95_records)*100:.1f}%)")

# 保存分析报告
with open(ANALYSIS_REPORT, 'w') as f:
    f.write("ASTRAL 95数据集分析报告\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"总序列数: {len(astral95_records)}\n")
    f.write(f"长度范围: {min(lengths)}-{max(lengths)} 残基\n")
    f.write(f"平均长度: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}\n")
    f.write(f"中位数长度: {np.median(lengths):.1f}\n\n")

    f.write("SCOP分类分布:\n")
    for scop_class, count in scop_counts.most_common():
        f.write(f"  {scop_class}: {count} ({count/len(astral95_records)*100:.1f}%)\n")

    f.write("\n氨基酸组成 (前10):\n")
    total_aa = sum(aa_composition.values())
    for aa, count in aa_composition.most_common(10):
        f.write(f"  {aa}: {count/total_aa*100:.2f}%\n")

print(f"✓ 分析报告已保存: {ANALYSIS_REPORT}")

# ============================================================================
# 步骤2: 加载现有序列并计算k-mer profiles
# ============================================================================
print("\n[步骤2] 加载现有1706条序列...")

existing_seqs = {}
for record in SeqIO.parse(EXISTING_FASTA, "fasta"):
    existing_seqs[record.id] = str(record.seq).upper()

print(f"✓ 已加载 {len(existing_seqs)} 条现有序列")

# 计算现有序列的k-mer profiles（采样加速）
print(f"✓ 计算现有序列的k-mer profiles (k={KMER_SIZE})...")
existing_profiles = {}
sample_size = min(500, len(existing_seqs))  # 采样500条用于去冗余检查
sampled_ids = np.random.choice(list(existing_seqs.keys()), sample_size, replace=False)
for seq_id in sampled_ids:
    existing_profiles[seq_id] = kmer_profile(existing_seqs[seq_id], k=KMER_SIZE)

print(f"✓ 已计算 {len(existing_profiles)} 条序列的k-mer profiles（采样）")

# ============================================================================
# 步骤3: 筛选ASTRAL 95序列
# ============================================================================
print(f"\n[步骤3] 从ASTRAL 95筛选补充序列（目标总数: {TARGET_TOTAL}）...")

target_supplement = TARGET_TOTAL - len(existing_seqs)
print(f"✓ 需要补充: {target_supplement} 条序列")

filtered_records = []
filter_stats = {
    'total': len(astral95_records),
    'length_pass': 0,
    'aa_composition_pass': 0,
    'redundancy_pass': 0,
    'final': 0
}

# 按SCOP类别分组，确保均衡抽取
scop_groups = defaultdict(list)
for record in astral95_records:
    scop_groups[record['scop_class']].append(record)

# 计算每个SCOP类别需要抽取的数量（均衡策略）
major_classes = ['all-alpha', 'all-beta', 'alpha/beta', 'alpha+beta']
samples_per_class = target_supplement // len(major_classes)

print(f"✓ 均衡抽取策略: 每个主要SCOP类别抽取 {samples_per_class} 条")

for scop_class in major_classes:
    print(f"\n  处理 {scop_class} 类别...")
    candidates = scop_groups[scop_class]
    np.random.shuffle(candidates)  # 随机打乱

    class_filtered = []
    for record in candidates:
        if len(class_filtered) >= samples_per_class:
            break

        seq = record['seq']
        length = record['length']

        # 长度筛选
        if not (MIN_LENGTH <= length <= MAX_LENGTH):
            continue
        filter_stats['length_pass'] += 1

        # 氨基酸组成筛选
        if not check_aa_composition(seq):
            continue
        filter_stats['aa_composition_pass'] += 1

        # 去冗余检查（与现有序列采样比较）
        seq_profile = kmer_profile(seq, k=KMER_SIZE)
        is_redundant = False

        # 随机采样200条现有序列进行比较（加速）
        sample_existing = np.random.choice(list(existing_profiles.keys()),
                                          min(200, len(existing_profiles)),
                                          replace=False)
        for existing_id in sample_existing:
            sim = kmer_similarity(seq_profile, existing_profiles[existing_id])
            if sim > SIMILARITY_THRESHOLD:
                is_redundant = True
                break

        if is_redundant:
            continue
        filter_stats['redundancy_pass'] += 1

        # 通过所有筛选
        class_filtered.append(record)
        # 添加到existing_profiles以避免内部冗余
        existing_profiles[record['id']] = seq_profile

    filtered_records.extend(class_filtered)
    print(f"  ✓ {scop_class}: 筛选出 {len(class_filtered)} 条序列")

filter_stats['final'] = len(filtered_records)

print(f"\n✓ 筛选完成:")
print(f"  - 总候选: {filter_stats['total']}")
print(f"  - 长度筛选通过: {filter_stats['length_pass']}")
print(f"  - 氨基酸组成通过: {filter_stats['aa_composition_pass']}")
print(f"  - 去冗余通过: {filter_stats['redundancy_pass']}")
print(f"  - 最终筛选: {filter_stats['final']}")

# ============================================================================
# 步骤4: 保存补充序列
# ============================================================================
print(f"\n[步骤4] 保存补充序列...")

# 保存FASTA
with open(SUPPLEMENT_FASTA, 'w') as f:
    for i, record in enumerate(filtered_records):
        seq_id = f"astral95|{record['scop_class']}|{record['id']}|{record['length']}"
        f.write(f">{seq_id}\n")
        f.write(f"{record['seq']}\n")

print(f"✓ FASTA已保存: {SUPPLEMENT_FASTA}")

# 保存元数据
metadata_rows = []
for i, record in enumerate(filtered_records):
    seq_id = f"astral95|{record['scop_class']}|{record['id']}|{record['length']}"
    metadata_rows.append({
        'seq_id': seq_id,
        'index': len(existing_seqs) + i,
        'category': 'astral95',
        'subcategory': record['scop_class'],
        'source_id': record['id'],
        'length': record['length'],
        'source': 'ASTRAL_95',
        'description': record['description']
    })

supplement_df = pd.DataFrame(metadata_rows)
supplement_df.to_csv(SUPPLEMENT_METADATA, index=False)
print(f"✓ 元数据已保存: {SUPPLEMENT_METADATA}")

# ============================================================================
# 步骤5: 细粒度分析
# ============================================================================
print(f"\n[步骤5] 细粒度分析...")

# 加载现有元数据
existing_df = pd.read_csv(EXISTING_METADATA)

# 合并统计
total_sequences = len(existing_df) + len(supplement_df)
print(f"\n✓ 总数据量统计:")
print(f"  - 现有序列: {len(existing_df)}")
print(f"  - 新增序列: {len(supplement_df)}")
print(f"  - 总计: {total_sequences}")

# 按类别统计
print(f"\n✓ 按类别划分:")
existing_category_counts = existing_df['category'].value_counts()
supplement_category_counts = supplement_df['category'].value_counts()

all_categories = set(existing_category_counts.index) | set(supplement_category_counts.index)
for cat in sorted(all_categories):
    existing_count = existing_category_counts.get(cat, 0)
    supplement_count = supplement_category_counts.get(cat, 0)
    total_count = existing_count + supplement_count
    print(f"  - {cat}: {existing_count} + {supplement_count} = {total_count}")

# 按SCOP类别统计（仅astral95）
print(f"\n✓ 新增序列按SCOP类别划分:")
scop_counts_new = supplement_df['subcategory'].value_counts()
for scop_class, count in scop_counts_new.items():
    print(f"  - {scop_class}: {count}")

# 长度分布对比
print(f"\n✓ 长度分布对比:")
existing_lengths = existing_df['length'].values
supplement_lengths = supplement_df['length'].values
print(f"  - 现有序列: {existing_lengths.min()}-{existing_lengths.max()} 残基, "
      f"平均 {existing_lengths.mean():.1f} ± {existing_lengths.std():.1f}")
print(f"  - 新增序列: {supplement_lengths.min()}-{supplement_lengths.max()} 残基, "
      f"平均 {supplement_lengths.mean():.1f} ± {supplement_lengths.std():.1f}")

# 保存扩展报告
with open(EXPANSION_REPORT, 'w') as f:
    f.write("数据集扩展与细粒度分析报告\n")
    f.write("=" * 80 + "\n\n")

    f.write("1. 总数据量统计\n")
    f.write("-" * 80 + "\n")
    f.write(f"现有序列: {len(existing_df)}\n")
    f.write(f"新增序列: {len(supplement_df)}\n")
    f.write(f"总计: {total_sequences}\n")
    f.write(f"扩展倍数: {total_sequences / len(existing_df):.2f}x\n\n")

    f.write("2. 按类别划分\n")
    f.write("-" * 80 + "\n")
    for cat in sorted(all_categories):
        existing_count = existing_category_counts.get(cat, 0)
        supplement_count = supplement_category_counts.get(cat, 0)
        total_count = existing_count + supplement_count
        f.write(f"{cat}:\n")
        f.write(f"  现有: {existing_count}\n")
        f.write(f"  新增: {supplement_count}\n")
        f.write(f"  总计: {total_count}\n")
        f.write(f"  占比: {total_count/total_sequences*100:.1f}%\n\n")

    f.write("3. 新增序列按SCOP类别划分\n")
    f.write("-" * 80 + "\n")
    for scop_class, count in scop_counts_new.items():
        f.write(f"{scop_class}: {count} ({count/len(supplement_df)*100:.1f}%)\n")

    f.write("\n4. 长度分布对比\n")
    f.write("-" * 80 + "\n")
    f.write(f"现有序列:\n")
    f.write(f"  范围: {existing_lengths.min()}-{existing_lengths.max()} 残基\n")
    f.write(f"  平均: {existing_lengths.mean():.1f} ± {existing_lengths.std():.1f}\n")
    f.write(f"  中位数: {np.median(existing_lengths):.1f}\n\n")
    f.write(f"新增序列:\n")
    f.write(f"  范围: {supplement_lengths.min()}-{supplement_lengths.max()} 残基\n")
    f.write(f"  平均: {supplement_lengths.mean():.1f} ± {supplement_lengths.std():.1f}\n")
    f.write(f"  中位数: {np.median(supplement_lengths):.1f}\n\n")

    f.write("5. 筛选统计\n")
    f.write("-" * 80 + "\n")
    f.write(f"ASTRAL 95总序列: {filter_stats['total']}\n")
    f.write(f"长度筛选通过率: {filter_stats['length_pass']/filter_stats['total']*100:.1f}%\n")
    f.write(f"氨基酸组成通过率: {filter_stats['aa_composition_pass']/filter_stats['length_pass']*100:.1f}%\n")
    f.write(f"去冗余通过率: {filter_stats['redundancy_pass']/filter_stats['aa_composition_pass']*100:.1f}%\n")
    f.write(f"最终筛选率: {filter_stats['final']/filter_stats['total']*100:.1f}%\n\n")

    f.write("6. 存储预算估算\n")
    f.write("-" * 80 + "\n")
    embedding_size_mb = total_sequences * 2560 * 2 / 1024 / 1024  # FP16
    f.write(f"序列级嵌入 (FP16): {embedding_size_mb:.2f} MB\n")
    f.write(f"预计推理时间 (batch_size=8): {total_sequences * 166 / 12178 / 60:.1f} 分钟\n")

print(f"✓ 扩展报告已保存: {EXPANSION_REPORT}")

print("\n" + "=" * 80)
print("数据集扩展完成!")
print("=" * 80)
print(f"✓ 总序列数: {len(existing_df)} + {len(supplement_df)} = {total_sequences}")
print(f"✓ 扩展倍数: {total_sequences / len(existing_df):.2f}x")
print(f"✓ 预计嵌入存储: {embedding_size_mb:.2f} MB (FP16)")
print(f"✓ 下一步: 运行 17_merge_final_dataset.py 合并最终数据集")
