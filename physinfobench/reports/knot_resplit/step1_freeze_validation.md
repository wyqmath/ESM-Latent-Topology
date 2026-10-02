# KNOT-RESPLIT 步骤 1：补算产物冻结与校验（记录）

- 时间：2026-09-25 17:00–17:20 Asia/Shanghai
- 输入版本：reports/knot_resplit/inputs_checksums.json（9 项 sha256 全钉：确认表/链清单/序列映射/
  14 条无档册/knots.tsv 冻结表/现行 split_manifest/抽链脚本/集群作业脚本/核算脚本）
- 脚本：scripts/knot_binding_check.py（核算固化）、scripts/validate_knot_binding_freeze.py（校验器）

## 校验结果（ALL PASSED）

| 检查 | 结果 |
|---|---|
| F1 输入钉版 | 9 项 sha256 记账，二次运行零漂移 |
| F2 确认表完整性 | 3,854 行全解析；0 自对、0 重复对、0 解析失败 |
| F3 覆盖声明 | **结构检查覆盖 1,006/1,020 usable 链；14 条无结构链保持"未验证"标记（data/curated/knots_sequences_unavailable.tsv），不把缺失当作通过** |
| F4 口径重现 | min≥0.6 边=2,998；跨集合=679（conf↔dev 301 / dev↔hold 230 / conf↔hold 148）；(pdb,chain) 归一映射零未解析；无自对 |
| F5 划分基线 | split_qc run_ts=2026-09-24 03:27 未变；git status data/splits/ 干净——旧划分原封未动 |

## 去重与归一说明

- 无序对唯一性：确认表逐对 (sorted(a,b)) 查重，无重复；
- 链 ID 归一：US-align 输出 `文件名` 形如 `{pdb}_{chain}`，以 (pdb.lower(), chain) 键映射回
  knot_chain_list/knots.tsv record_id，再经 manifest 后缀原样+小写双键解析 split（P3.01 审计发现的
  大小写差异在此消解，0 未解析）；
- 集合映射基线=现行冻结 manifest（旧版），步骤 4 之前不生成任何新 manifest。

## 注（步骤 5 审核一审 m7）

validate_knot_binding_freeze.py 将步骤 1 时点状态钉死（inputs_checksums + split_qc run_ts +
data/splits 干净）；步骤 4 生成新版后此脚本**仅对步骤 1 时点可重跑**（git stash/checkout 至
c41bf6c 可复现），在当前工作区运行会因 F1/F5 报漂移——这是冻结校验的预期行为，不是缺陷。
