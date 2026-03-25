#!/usr/bin/env python3
# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
生成随机氨基酸序列作为对照组

输入: 无
输出:
  - data/random_sequences.fasta
  - data/random_metadata.csv

功能: 生成随机氨基酸序列，长度分布匹配真实蛋白质
"""

import random
from pathlib import Path

def generate_random_sequence(length: int, seed: int = None) -> str:
    """
    生成随机氨基酸序列
    使用天然蛋白质的氨基酸频率分布
    """
    # 天然蛋白质中的氨基酸频率（来自UniProt统计）
    # 参考: Brocchieri & Karlin, Proteins 2005
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    frequencies = [
        0.0825,  # A - Alanine
        0.0137,  # C - Cysteine
        0.0545,  # D - Aspartic acid
        0.0675,  # E - Glutamic acid
        0.0386,  # F - Phenylalanine
        0.0708,  # G - Glycine
        0.0227,  # H - Histidine
        0.0596,  # I - Isoleucine
        0.0966,  # K - Lysine
        0.0965,  # L - Leucine
        0.0242,  # M - Methionine
        0.0406,  # N - Asparagine
        0.0470,  # P - Proline
        0.0393,  # Q - Glutamine
        0.0553,  # R - Arginine
        0.0656,  # S - Serine
        0.0534,  # T - Threonine
        0.0687,  # V - Valine
        0.0108,  # W - Tryptophan
        0.0292,  # Y - Tyrosine
    ]

    if seed is not None:
        random.seed(seed)

    # 根据频率分布生成序列
    sequence = random.choices(amino_acids, weights=frequencies, k=length)
    return "".join(sequence)

def main():
    # 创建输出目录
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # 输出文件路径
    fasta_file = data_dir / "random_sequences.fasta"
    metadata_file = data_dir / "random_metadata.csv"

    sequences = []
    metadata = []

    print("生成随机氨基酸序列...")

    # 长度分布匹配真实蛋白质
    # 参考CullPDB数据集: 40-300残基，平均166±69
    # 使用正态分布生成长度
    random.seed(42)  # 固定种子以保证可重复性

    for i in range(500):
        # 生成长度（正态分布，均值166，标准差69，范围40-300）
        length = int(random.gauss(166, 69))
        length = max(40, min(300, length))  # 限制在40-300范围内

        # 生成随机序列
        sequence = generate_random_sequence(length, seed=42 + i)

        # 序列ID格式: control|random|id|length
        seq_id = f"control|random|random_{i+1:04d}|{length}"

        sequences.append((seq_id, sequence))
        metadata.append({
            "seq_id": seq_id,
            "sequence_id": f"random_{i+1:04d}",
            "length": length,
            "category": "control",
            "subcategory": "random",
            "generation_method": "random_with_natural_frequencies",
        })

    # 写入FASTA文件
    with open(fasta_file, 'w') as f:
        for seq_id, sequence in sequences:
            f.write(f">{seq_id}\n")
            # 每行60个字符
            for i in range(0, len(sequence), 60):
                f.write(sequence[i:i+60] + "\n")

    print(f"✓ FASTA文件已保存: {fasta_file}")
    print(f"  总序列数: {len(sequences)}")

    # 写入元数据CSV
    import csv
    with open(metadata_file, 'w', newline='') as f:
        fieldnames = ["seq_id", "sequence_id", "length", "category", "subcategory", "generation_method"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)

    print(f"✓ 元数据已保存: {metadata_file}")

    # 统计信息
    print("\n=== 统计信息 ===")
    print(f"总序列数: {len(sequences)}")

    lengths = [m["length"] for m in metadata]
    print(f"长度范围: {min(lengths)}-{max(lengths)} 残基")
    print(f"平均长度: {sum(lengths)/len(lengths):.1f}±{(sum((l-sum(lengths)/len(lengths))**2 for l in lengths)/len(lengths))**0.5:.1f}")

    # 氨基酸组成分析
    all_seq = "".join([s[1] for s in sequences])
    aa_counts = {}
    for aa in all_seq:
        aa_counts[aa] = aa_counts.get(aa, 0) + 1

    print("\n氨基酸组成 (应接近天然频率):")
    expected_freq = {
        'A': 8.25, 'C': 1.37, 'D': 5.45, 'E': 6.75, 'F': 3.86,
        'G': 7.08, 'H': 2.27, 'I': 5.96, 'K': 9.66, 'L': 9.65,
        'M': 2.42, 'N': 4.06, 'P': 4.70, 'Q': 3.93, 'R': 5.53,
        'S': 6.56, 'T': 5.34, 'V': 6.87, 'W': 1.08, 'Y': 2.92,
    }

    print(f"{'AA':<4} {'Observed':<10} {'Expected':<10} {'Diff':<10}")
    print("-" * 40)
    for aa in sorted(aa_counts.keys()):
        obs_freq = aa_counts[aa] / len(all_seq) * 100
        exp_freq = expected_freq.get(aa, 0)
        diff = obs_freq - exp_freq
        print(f"{aa:<4} {obs_freq:>6.2f}%    {exp_freq:>6.2f}%    {diff:>+6.2f}%")

    print("\n随机序列生成完成，可作为对照组")

if __name__ == "__main__":
    main()
