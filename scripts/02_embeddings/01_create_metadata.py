#!/usr/bin/env python3
# -- Project root setup --
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])
"""
创建统一的元数据表格，整合所有序列的信息和物理指标

输入文件:
- data/all_metadata.csv (合并的元数据)
- data/pdb_files/*.pdb (PDB结构文件，用于计算物理指标)
- data/physical_indicators/*.csv (已有的物理指标结果)

输出文件:
- data/metadata_complete.csv (完整元数据表格)

功能描述:
1. 加载合并的元数据
2. 为856个anchor序列关联1.0项目的物理指标(E[n], condition_number, wilson_loop)
3. 为其他序列填充占位符(NaN)
4. 生成统一的CSV表格，包含所有必要字段
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# 设置路径
data_dir = Path("data")
test_dir = Path("data/raw")  # 原 1.0/test，legacy 数据已迁移

print("=" * 80)
print("创建完整元数据表格")
print("=" * 80)

# 1. 加载合并的元数据
print("\n[1/4] 加载合并的元数据...")
all_metadata_path = data_dir / "all_metadata.csv"
if not all_metadata_path.exists():
    raise FileNotFoundError(f"未找到合并的元数据文件: {all_metadata_path}")

df = pd.read_csv(all_metadata_path)
print(f"  ✓ 加载 {len(df)} 条序列的元数据")
print(f"  ✓ 现有字段: {list(df.columns)}")

# 2. 初始化物理指标列
print("\n[2/4] 初始化物理指标列...")
df['E_n'] = np.nan
df['condition_number'] = np.nan
df['wilson_loop'] = np.nan
df['scop_class'] = ''
print("  ✓ 添加物理指标列: E_n, condition_number, wilson_loop, scop_class")

# 3. 为anchor序列关联物理指标
print("\n[3/4] 关联anchor序列的物理指标...")

# 尝试从1.0项目加载已有的物理指标结果
anchor_metadata_path = data_dir / "anchor_metadata.csv"
if anchor_metadata_path.exists():
    anchor_df = pd.read_csv(anchor_metadata_path)
    print(f"  ✓ 加载anchor元数据: {len(anchor_df)} 条")

    # 创建PDB ID到物理指标的映射
    # anchor序列的seq_id格式: anchor|scop_class|pdb_id|length
    for idx, row in df[df['category'] == 'anchor'].iterrows():
        seq_id = row['seq_id']
        parts = seq_id.split('|')
        if len(parts) >= 3:
            pdb_id = parts[2]

            # 在anchor_df中查找对应的物理指标
            anchor_row = anchor_df[anchor_df['seq_id'] == seq_id]
            if not anchor_row.empty:
                df.at[idx, 'E_n'] = anchor_row.iloc[0]['E_n']
                df.at[idx, 'condition_number'] = anchor_row.iloc[0].get('condition_number', np.nan)
                df.at[idx, 'wilson_loop'] = anchor_row.iloc[0].get('wilson_loop', np.nan)
                df.at[idx, 'scop_class'] = parts[1]  # 从seq_id提取SCOP类别

    anchor_with_E_n = df[(df['category'] == 'anchor') & (~df['E_n'].isna())]
    print(f"  ✓ 成功关联 {len(anchor_with_E_n)} 条anchor序列的物理指标")
else:
    print(f"  ⚠ 未找到anchor元数据文件: {anchor_metadata_path}")
    print("  ⚠ 跳过物理指标关联")

# 4. 生成统计摘要
print("\n[4/4] 生成统计摘要...")
print("\n类别分布:")
print(df['category'].value_counts())

print("\n物理指标统计 (仅anchor序列):")
anchor_df_filtered = df[df['category'] == 'anchor']
if not anchor_df_filtered['E_n'].isna().all():
    print(f"  E[n]: {anchor_df_filtered['E_n'].describe()}")
    print(f"\n  SCOP类别分布:")
    print(anchor_df_filtered['scop_class'].value_counts())
else:
    print("  ⚠ 无可用的物理指标数据")

# 5. 保存完整元数据表格
output_path = data_dir / "metadata_complete.csv"
df.to_csv(output_path, index=False)
print(f"\n✓ 完整元数据表格已保存: {output_path}")
print(f"  总序列数: {len(df)}")
print(f"  字段数: {len(df.columns)}")
print(f"  字段列表: {list(df.columns)}")

# 6. 生成简要报告
report_path = data_dir / "metadata_report.txt"
with open(report_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("完整元数据表格报告\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"总序列数: {len(df)}\n")
    f.write(f"字段数: {len(df.columns)}\n\n")

    f.write("类别分布:\n")
    f.write(df['category'].value_counts().to_string() + "\n\n")

    f.write("长度统计:\n")
    f.write(df.groupby('category')['length'].describe().to_string() + "\n\n")

    if not anchor_df_filtered['E_n'].isna().all():
        f.write("物理指标统计 (anchor序列):\n")
        f.write(f"  E[n]: {anchor_df_filtered['E_n'].describe().to_string()}\n\n")
        f.write("  SCOP类别分布:\n")
        f.write(anchor_df_filtered['scop_class'].value_counts().to_string() + "\n\n")

    f.write("字段列表:\n")
    for i, col in enumerate(df.columns, 1):
        f.write(f"  {i}. {col}\n")

print(f"✓ 元数据报告已保存: {report_path}")

print("\n" + "=" * 80)
print("任务完成!")
print("=" * 80)
