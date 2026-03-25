#!/usr/bin/env python3
# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
合并所有序列到统一FASTA文件

输入:
  - data/anchor_sequences.fasta (856条)
  - data/integrable_island_sequences.fasta (50条)
  - data/knotted_286_full_length.fasta (286条)
  - data/fold_switching_paper_full_length.fasta (79条)
  - data/idp_1000_full_length.fasta (1000条)
  - data/random_sequences.fasta (500条)

输出:
  - data/all_sequences.fasta (总计1706条)
  - data/all_metadata.csv (合并元数据)

功能: 合并所有数据集，添加统一索引，生成最终数据集
"""

from pathlib import Path
import csv
from Bio import SeqIO

def load_fasta(fasta_file):
    """加载FASTA文件"""
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append((record.id, str(record.seq)))
    return sequences

def load_metadata(csv_file):
    """加载元数据CSV"""
    metadata = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        metadata = list(reader)
    return metadata

def main():
    data_dir = Path("data")

    # 输入文件列表
    input_files = [
        ("anchor_sequences.fasta", "anchor_metadata.csv", "anchor"),
        ("integrable_island_sequences.fasta", "integrable_island_metadata.csv", "integrable"),
        ("knotted_286_full_length.fasta", "knotted_286_metadata.csv", "knotted"),
        ("fold_switching_paper_full_length.fasta", "fold_switching_metadata.csv", "fold_switching"),
        ("idp_1000_full_length.fasta", "idp_1000_metadata.csv", "idp"),
        ("random_sequences.fasta", "random_metadata.csv", "random"),
    ]

    all_sequences = []
    all_metadata = []

    print("合并所有序列文件...")

    for fasta_name, csv_name, dataset_name in input_files:
        fasta_path = data_dir / fasta_name
        csv_path = data_dir / csv_name

        if not fasta_path.exists():
            print(f"⚠ 警告: {fasta_path} 不存在，跳过")
            continue

        # 加载序列
        sequences = load_fasta(fasta_path)
        print(f"  {dataset_name}: {len(sequences)} 条序列")

        # 加载元数据
        metadata = []
        if csv_path.exists():
            metadata = load_metadata(csv_path)

        # 添加到总列表
        all_sequences.extend(sequences)
        all_metadata.extend(metadata)

    print(f"\n总计: {len(all_sequences)} 条序列")

    # 写入合并后的FASTA文件
    output_fasta = data_dir / "all_sequences.fasta"
    with open(output_fasta, 'w') as f:
        for seq_id, sequence in all_sequences:
            f.write(f">{seq_id}\n")
            # 每行60个字符
            for i in range(0, len(sequence), 60):
                f.write(sequence[i:i+60] + "\n")

    print(f"✓ 合并FASTA文件已保存: {output_fasta}")

    # 合并元数据
    # 统一字段名
    unified_metadata = []
    for i, (seq_id, sequence) in enumerate(all_sequences):
        # 从seq_id解析信息
        parts = seq_id.split('|')
        category = parts[0] if len(parts) > 0 else "unknown"
        subcategory = parts[1] if len(parts) > 1 else "unknown"
        length = len(sequence)

        # 查找对应的元数据
        meta = None
        for m in all_metadata:
            if m.get("seq_id") == seq_id:
                meta = m
                break

        # 构建统一元数据
        unified_meta = {
            "index": i,
            "seq_id": seq_id,
            "category": category,
            "subcategory": subcategory,
            "length": length,
        }

        # 添加额外字段
        if meta:
            for key, value in meta.items():
                if key not in unified_meta:
                    unified_meta[key] = value

        unified_metadata.append(unified_meta)

    # 写入合并后的元数据
    output_csv = data_dir / "all_metadata.csv"

    # 收集所有字段名
    all_fields = set()
    for meta in unified_metadata:
        all_fields.update(meta.keys())

    # 固定顺序的核心字段
    core_fields = ["index", "seq_id", "category", "subcategory", "length"]
    other_fields = sorted(all_fields - set(core_fields))
    fieldnames = core_fields + other_fields

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for meta in unified_metadata:
            writer.writerow(meta)

    print(f"✓ 合并元数据已保存: {output_csv}")

    # 统计信息
    print("\n=== 数据集统计 ===")
    print(f"总序列数: {len(all_sequences)}")

    # 按类别统计
    category_counts = {}
    for meta in unified_metadata:
        cat = meta["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print("\n按类别统计:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")

    # 按子类别统计
    subcategory_counts = {}
    for meta in unified_metadata:
        subcat = meta["subcategory"]
        subcategory_counts[subcat] = subcategory_counts.get(subcat, 0) + 1

    print("\n按子类别统计:")
    for subcat, count in sorted(subcategory_counts.items(), key=lambda x: -x[1]):
        print(f"  {subcat}: {count}")

    # 长度统计
    lengths = [meta["length"] for meta in unified_metadata]
    print(f"\n长度统计:")
    print(f"  范围: {min(lengths)}-{max(lengths)} 残基")
    print(f"  平均: {sum(lengths)/len(lengths):.1f}±{(sum((l-sum(lengths)/len(lengths))**2 for l in lengths)/len(lengths))**0.5:.1f}")

    # 估算存储空间
    total_residues = sum(lengths)
    fp16_size_residue = 2560 * 2  # 2560维 × 2字节(FP16)
    fp16_size_sequence = 2560 * 2  # 序列级嵌入

    estimated_residue_level = total_residues * fp16_size_residue / (1024**3)  # GB
    estimated_sequence_level = len(all_sequences) * fp16_size_sequence / (1024**3)  # GB

    print(f"\n存储空间估算 (FP16):")
    print(f"  残基级嵌入 (N×2560): ~{estimated_residue_level:.1f} GB")
    print(f"  序列级嵌入 (1×2560): ~{estimated_sequence_level:.2f} GB")

    print("\n✓ 数据集合并完成")

if __name__ == "__main__":
    main()
