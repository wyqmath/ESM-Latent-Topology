# KNOT-RESPLIT 步骤 4：新版划分生成、审计与受影响重跑（已完成）

- 时间：2026-09-25 17:30–18:56 Asia/Shanghai（18:56 为 BLOCKER 修复后确定性重生成时点）
- 生成脚本：scripts/knot_resplit_finalize.py（mode-0 等价性检查内建于生成流程；bg 写出经
  gzip mtime=0 确定性修复，**修复后两轮完整运行 checksums 全同**（一审 BLOCKER-1 指出修复前
  "三轮全同"系空转，已更正并留档 step4_determinism_record.md））
- 旧版留档：reports/knot_resplit/old_version/（manifest/split_qc/checksums/bg + sha256）

## 1. 新版划分（已写入 data/splits/，17:07 生成）

- split_manifest.tsv（4,858 行；sha256 41676fd7…）
- fs_l2_background_manifest.tsv.gz（sha256 **a4b9fea6**…，mtime=0 确定性封装；内容与交付版逐字节相同）
- knot_structural_edges.tsv（2,998 条打结结构绑定边，冻结入档）
- split_qc.json（groups_total 68,875；prev_version=69,084@03:27；forced_dev 全 dev=true；
  knots presence 可评价：conf 24 阳/126 阴、hold 26 阳/83 阴；type：conf 仅 3_1×16、
  hold 3_1 18/4_1 1/5_1 1、dev 76）
- checksums.txt（三文件 sha256）

## 2. 审计（scripts/audit_leakage.py 扩展 L6 后重跑）

L1–L5 全零（同 P2.03 口径）；**新增 L6：打结结构绑定跨集合 = 0/2,998 ✓**；
如实披露：L4 FS max 单侧跨集合 55（旧 31——重划分后单侧相似重排）、
L4b 打结 max 单侧跨集合 253（信息性）、14 条无结构链保持未验证。

## 3. 逐样本差异（reports/knot_resplit/step4_manifest_diff.tsv）

- 受影响 4,110/4,858（bpti 9 / disorder 2,792 / knot 1,234 / fold_switch 75）
- 迁移审计：S1/S2 曝光样本迁出 dev = **0** ✓；conf→dev 511、hold→dev 473
  （并入曝光/绑定根组）、dev→conf 445、dev→hold 456、conf↔hold 182
- 结论：**全部任务受影响**（印证用户"不预设 FS/disorder/L2 不受影响"），按下节重跑。

## 4. 重跑清单与状态

| 项 | 状态 |
|---|---|
| 输入重建（build_p303_inputs.py 指向新 manifest） | ✓ 完成（knots dev usable 750、disorder 提取 2,279+34 空域+79 全域超界、stage-A 重抽样 sha d0d3d07f…） |
| 增量嵌入（集群 813037，GPU） | ✓ 完成（knots 750 / fs_region 16 / disorder 2,279 / stage-A 10,000 / stage-B 增量 2,462） |
| 全任务探针重跑（集群 813047/813048） | ✓ 完成 |
| L2 stage-A 重选 + stage-B 重打分 | ✓ 完成（813048/813051） |
| L2 元数据基线 B0–B3 重跑（新 dev 宇宙 342,014） | ✓ 完成（813053） |
| L2 传播 qc（簇大小分布等） | 随 stage-B 重打分更新 |
| 全任务探针结果 | 见下节 | |

## 5. 边界

- 旧版划分及全部旧结果保留于 reports/knot_resplit/old_version/ 与 git 历史；不删除不改写。
- 确认/保留标签读取权限不变：本步骤零新增标签消费（探针/打分/基线均只读 ID+结构+已登记约束）。


## 探针重跑结果（新划分；development 内 CV 点估计；18:21 完成）

| 任务 | 指标 | 选中层 | C | CV 均值 | ±std | n |
|---|---|---|---|---|---|---|
| T-KNOT-PRESENCE | AUROC | 6 | 1 | 0.929 | 0.023 | 750 |
| T-KNOT-TYPE | Macro-F1 | 6 | 0.01 | 0.506 | 0.079 | 71（=76 可评口径−5 无序列链） |
| T-FS-REGION 主分析 | residue AUPRC | 3 | 0.01 | 0.105 | 0.068 | 3 对 6 链 |
| T-FS-REGION 敏感性 | residue AUPRC | 3 | 0.01 | 0.214 | 0.214 | 6 对 10 链 |
| T-DISORDER-RES | residue AUPRC | 3 | 0.01 | 0.904 | 0.049 | 2,279 |
| T-FS-L2 stage-A 选择 | rank percentile | 6 | 0.01 | 0.00169 | - | 10,010 |

## L2 官方指标（新划分 stage-B；传播语义披露同前）

- 宇宙 342,014 = 背景 342,004（新版 dev，动态校验）+ 10 strict pair 单元；选中 L33/C0.01。
- **recall@10=0.2、recall@100=0.5、recall@1000=1.0、recall@10000=1.0；percentile=0.00031**
  （pos_ranks=2/9/13/92/93/131/147/148/185/239）。
- 基线重跑（同宇宙）：B0 percentile 0.440 / B1 0.579 / B2 0.353 / B3 0.613；recall@10000 全部 ≤0.1
  ——PLM 表示打分器大幅高于全部元数据基线；直推乐观语义与传播语义照例披露。
- 传播 qc：同簇同分为构造性性质；**背景簇大小 top=1,019**（propagation_qc 口径；1,521 为
  含背景成员的 labeled 根最大成员数——两个口径，前者为本句所指）。
