#!/usr/bin/env python3
# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
提取 Paper 1.0 中的 856 个 PDB 锚点序列

输入文件:
  - data/physical_indicators/physical_indicators.csv: 856条PDB的物理指标数据
  - data/pdb_files/*.pdb: PDB结构文件

输出文件:
  - data/anchor_sequences.fasta: 856条序列的FASTA文件
  - data/anchor_metadata.csv: 元数据表格（包含物理指标）

功能描述:
  从1.0项目提取856个PDB序列，关联已计算的物理指标（E[n]、条件数、V_re等），
  为后续ESM-2特征提取和流形分析提供基准锚点。
"""

import json
import os
from pathlib import Path
import pandas as pd
from Bio import PDB
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import numpy as np

# 路径配置 (chdir 已切换到项目根目录，以下均为相对路径)
RESULTS_JSON = Path("data/physical_indicators/physical_indicators.csv")  # 原 legacy_1.0/results_all.json，已提取为 CSV
PDB_DIR = Path("data/pdb_files")
OUTPUT_DIR = Path("data")
OUTPUT_FASTA = OUTPUT_DIR / "anchor_sequences.fasta"
OUTPUT_CSV = OUTPUT_DIR / "anchor_metadata.csv"

# 创建输出目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 氨基酸三字母到单字母映射
AA_3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def extract_sequence_from_pdb(pdb_file, chain_id):
    """从PDB文件提取指定链的序列"""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                sequence = []
                for residue in chain:
                    if residue.id[0] == ' ':  # 标准残基
                        resname = residue.resname
                        if resname in AA_3TO1:
                            sequence.append(AA_3TO1[resname])
                return ''.join(sequence)
    return None

def calculate_integrability_error(V_re, V_im, kappa_deg, tau_deg):
    """
    计算可积性误差 E[n]

    根据 Paper 1.0/2.0 的公式:
    E[n] = |V_eff[n] - V_re[n]| / |V_eff[n]|
    其中 V_eff = κ² + τ²

    注意: V_re/V_im 长度为 N-2 (因为需要3点计算曲率)
          kappa/tau 长度为 N-1 或 N (取决于计算方式)
          需要对齐长度
    """
    if not V_re or not V_im or not kappa_deg or not tau_deg:
        return None

    # 转换为numpy数组
    V_re = np.array(V_re)
    V_im = np.array(V_im)
    kappa = np.deg2rad(kappa_deg)
    tau = np.deg2rad(tau_deg)

    # 对齐长度: 取最短长度
    min_len = min(len(V_re), len(kappa), len(tau))
    V_re = V_re[:min_len]
    kappa = kappa[:min_len]
    tau = tau[:min_len]

    # 计算 V_eff
    V_eff = kappa**2 + tau**2

    # 计算误差
    errors = np.abs(V_eff - V_re) / (np.abs(V_eff) + 1e-10)
    mean_error = np.mean(errors)

    return mean_error

def main():
    print("=" * 60)
    print("提取 Paper 1.0 的 856 个 PDB 锚点序列")
    print("=" * 60)

    # 1. 加载 results_all.json
    print(f"\n[1/4] 加载物理指标数据: {RESULTS_JSON}")
    with open(RESULTS_JSON, 'r') as f:
        results = json.load(f)
    print(f"  ✓ 加载了 {len(results)} 条记录")

    # 2. 提取序列
    print(f"\n[2/4] 从 PDB 文件提取序列: {PDB_DIR}")
    sequences = []
    metadata_records = []
    failed = []

    for i, entry in enumerate(results):
        pdb_id = entry['pdb_id']
        chain = entry['chain']
        chain_id = entry['chain_id']

        # 构建PDB文件路径
        pdb_file = PDB_DIR / f"{pdb_id}.pdb"

        if not pdb_file.exists():
            failed.append(chain_id)
            continue

        # 提取序列
        seq = extract_sequence_from_pdb(pdb_file, chain)
        if seq is None:
            failed.append(chain_id)
            continue

        # 计算可积性误差
        E_n = calculate_integrability_error(
            entry.get('V_re'),
            entry.get('V_im'),
            entry.get('kappa_deg'),
            entry.get('tau_deg')
        )

        # 创建FASTA记录
        seq_id = f"anchor|pdb|{chain_id}|{len(seq)}"
        record = SeqRecord(
            Seq(seq),
            id=seq_id,
            description=f"scop_class={entry.get('scop_class', 'NA')} resolution={entry.get('resolution', 'NA')}"
        )
        sequences.append(record)

        # 创建元数据记录
        metadata = {
            'seq_id': seq_id,
            'category': 'anchor',
            'subcategory': 'pdb',
            'source_id': chain_id,
            'length': len(seq),
            'scop_class': entry.get('scop_class', 'NA'),
            'scop_label': entry.get('label', 'NA'),
            'resolution': entry.get('resolution', None),
            'E_n': E_n,
            'mean_vim': entry.get('mean_vim', None),
            'mean_vre': entry.get('mean_vre', None),
            'vim_vre_ratio': entry.get('vim_vre_ratio', None),
            'geom_corr': entry.get('geom_corr', None),
            'vre_rmsd': entry.get('vre_rmsd', None),
            'n_helix': entry.get('n_H', None),
            'n_sheet': entry.get('n_E', None),
            'n_coil': entry.get('n_C', None),
        }
        metadata_records.append(metadata)

        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{len(results)}")

    print(f"  ✓ 成功提取 {len(sequences)} 条序列")
    if failed:
        print(f"  ⚠ 失败 {len(failed)} 条: {failed[:5]}...")

    # 3. 保存FASTA文件
    print(f"\n[3/4] 保存 FASTA 文件: {OUTPUT_FASTA}")
    SeqIO.write(sequences, OUTPUT_FASTA, "fasta")
    print(f"  ✓ 已保存 {len(sequences)} 条序列")

    # 4. 保存元数据CSV
    print(f"\n[4/4] 保存元数据表格: {OUTPUT_CSV}")
    df = pd.DataFrame(metadata_records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  ✓ 已保存 {len(df)} 条记录")

    # 统计信息
    print("\n" + "=" * 60)
    print("统计信息")
    print("=" * 60)
    print(f"总序列数: {len(sequences)}")
    print(f"长度范围: {df['length'].min()}-{df['length'].max()} 残基")
    print(f"平均长度: {df['length'].mean():.1f} ± {df['length'].std():.1f} 残基")
    print(f"\nSCOP 类别分布:")
    print(df['scop_label'].value_counts())
    print(f"\n可积性误差 E[n] 统计:")
    print(f"  均值: {df['E_n'].mean():.4f}")
    print(f"  中位数: {df['E_n'].median():.4f}")
    print(f"  范围: {df['E_n'].min():.4f} - {df['E_n'].max():.4f}")

    print("\n✓ 任务完成！")

if __name__ == "__main__":
    main()
