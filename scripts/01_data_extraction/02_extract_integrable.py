#!/usr/bin/env python3
# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
提取可积岛序列（纯螺旋肽）

输入: data/helical_dataset/*.pdb (100条螺旋肽PDB文件)
输出:
  - data/integrable_island_sequences.fasta (50条序列)
  - data/integrable_island_metadata.csv (元数据)

功能: 从2.0项目的螺旋肽数据集中提取序列，选择50条作为可积岛探针
"""

import os
import glob
import pandas as pd
from Bio import PDB
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import warnings
warnings.filterwarnings('ignore')

# 路径配置
HELICAL_DATASET_DIR = "data/helical_dataset"
OUTPUT_DIR = "data"
OUTPUT_FASTA = os.path.join(OUTPUT_DIR, "integrable_island_sequences.fasta")
OUTPUT_METADATA = os.path.join(OUTPUT_DIR, "integrable_island_metadata.csv")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_sequence_from_pdb(pdb_file):
    """从PDB文件提取序列"""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    # 标准氨基酸三字母到单字母映射
    aa_map = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }

    sequences = []
    for model in structure:
        for chain in model:
            seq = []
            for residue in chain:
                if residue.id[0] == ' ':  # 标准残基
                    resname = residue.resname
                    if resname in aa_map:
                        seq.append(aa_map[resname])
            if seq:
                sequences.append(''.join(seq))

    return sequences[0] if sequences else None

def main():
    print("=" * 60)
    print("提取可积岛序列（纯螺旋肽）")
    print("=" * 60)

    # 获取所有PDB文件
    pdb_files = sorted(glob.glob(os.path.join(HELICAL_DATASET_DIR, "*.pdb")))
    print(f"\n找到 {len(pdb_files)} 个螺旋肽PDB文件")

    # 提取序列
    records = []
    metadata = []

    for i, pdb_file in enumerate(pdb_files):
        pdb_id = os.path.basename(pdb_file).replace('.pdb', '')

        try:
            sequence = extract_sequence_from_pdb(pdb_file)
            if sequence:
                seq_length = len(sequence)

                # 创建序列ID: integrable|helix|pdb_id|length
                seq_id = f"integrable|helix|{pdb_id}|{seq_length}"

                # 创建SeqRecord
                record = SeqRecord(
                    Seq(sequence),
                    id=seq_id,
                    description=f"Helical peptide from 2.0 project"
                )
                records.append(record)

                # 记录元数据
                metadata.append({
                    'seq_id': seq_id,
                    'category': 'integrable',
                    'subcategory': 'helix',
                    'pdb_id': pdb_id,
                    'length': seq_length,
                    'source': '2.0_helical_dataset',
                    'description': 'Pure helical peptide (integrable island)'
                })

                if (i + 1) % 20 == 0:
                    print(f"  已处理 {i + 1}/{len(pdb_files)} 个文件...")

        except Exception as e:
            print(f"  警告: 无法处理 {pdb_id}: {e}")
            continue

    print(f"\n成功提取 {len(records)} 条序列")

    # 选择50条序列（如果超过50条，均匀采样）
    if len(records) > 50:
        print(f"从 {len(records)} 条中选择 50 条...")
        # 均匀采样
        indices = [int(i * len(records) / 50) for i in range(50)]
        selected_records = [records[i] for i in indices]
        selected_metadata = [metadata[i] for i in indices]
    else:
        selected_records = records
        selected_metadata = metadata

    # 保存FASTA文件
    SeqIO.write(selected_records, OUTPUT_FASTA, "fasta")
    print(f"\n已保存 {len(selected_records)} 条序列到: {OUTPUT_FASTA}")

    # 保存元数据
    df = pd.DataFrame(selected_metadata)
    df.to_csv(OUTPUT_METADATA, index=False)
    print(f"已保存元数据到: {OUTPUT_METADATA}")

    # 统计信息
    print("\n" + "=" * 60)
    print("统计信息")
    print("=" * 60)
    print(f"序列数量: {len(selected_records)}")
    print(f"长度范围: {df['length'].min()}-{df['length'].max()} 残基")
    print(f"平均长度: {df['length'].mean():.1f} ± {df['length'].std():.1f} 残基")
    print(f"中位数长度: {df['length'].median():.0f} 残基")

    # 显示前5条序列
    print("\n前5条序列:")
    for i, record in enumerate(selected_records[:5]):
        print(f"  {i+1}. {record.id}: {len(record.seq)} 残基")

    print("\n✓ 可积岛序列提取完成!")

if __name__ == "__main__":
    main()
