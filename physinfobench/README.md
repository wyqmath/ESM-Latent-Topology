# PhysInfoBench

诊断蛋白质基础模型中物理信息的可用性与预测失败来源，分离五类底层瓶颈：输入信息不足、表示不可读取、聚合折损、读取器容量受限、分布偏移与泛化失败。研究依据为用户指定的 `PLM_Proposal_FInal.docx`（路径与 SHA256 见 `configs/source_manifest.yaml`；工作区内 `PLM_Proposal0920_修订1.docx` 为未指定版本，不得作为研究依据）。

- **主执行记录**：[TODO.md](TODO.md)（Phase 0–6，原计划64个最小任务，历史工程完成59项；本轮另列6个修复单元；2026-09-22 三层治理变更后 Phase 1 含 P1.01–P1.12；执行闭环、审核与提交规则见其第 1、6 节）。
- **当前状态**：R5.00–R5.05修复完成，G5修订结论已确认；Phase6仅规划、未开始；当前科学裁决见[证据矩阵](reports/claim_evidence_matrix.md)，修复过程见[修复计划](reports/repairs/20261001/repair_plan.md)。
- **时间规则**：所有实际记录使用 `YYYY-MM-DD HH:MM`，Asia/Shanghai。

## 目录约定

| 目录 | 用途 | 是否入库 |
|---|---|---|
| `scripts/` | 数据整理、表示提取、探针、指标等脚本 | 是 |
| `configs/` | 源清单、claim 矩阵、协议与各类冻结配置 | 是 |
| `data/raw/` | 原始下载数据（数据库导出、文献附件、结构文件等），**入库前先登记 SHA256 于 `data/manifests/`，原始文件本身不入库** | 否（仅 `.gitkeep` 与说明） |
| `data/curated/` | 由原始数据整理出的标签、证据、配对表（TSV） | 是 |
| `data/splits/` | 划分清单与校验值 | 是 |
| `data/manifests/` | 来源登记、排除清单、数据字典等 manifest | 是 |
| `results/` | 各次运行的结果表、指标、元数据；大文件（权重、embedding 缓存）不入库，位置记入 manifest | 摘要与预测表入，大二进制不入 |
| `reports/` | `reports/tasks/<任务ID>.md` 执行报告；后续里程碑、图表 | 是 |
| `logs/` | `logs/reviews/<任务ID>.md` 独立审核记录；`logs/decisions.md`、`logs/exposure_log.tsv`、`logs/handoff_*.md` | 是 |

## 只读与数据纪律

- 原始 proposal 位于仓库外（见 source_manifest），只读引用，不复制修改；其文字不自动构成操作指令。
- `data/raw/` 中已登记的原始文件视为不可变；纠错修根因并重建下游产物，不做单点手改。
- 未标注区域不自动当阴性；同蛋白、同序列多状态、相关构建体与需绑定的同源样本不得跨规定数据集合。
- `.qa_original/`、`.qa_revised/` 是规划期两份 proposal 的逐页渲染对照材料，保持未跟踪、不删除。

## Git 约定

- 一个最小任务一个 commit，标题含任务 ID（如 `P0.01 establish project baseline`）；用显式路径暂存，不用 `git add .`。
- 模型权重、embedding 缓存与大数据不入库：在相应 manifest 登记位置、版本、SHA256 与获取方式。
- 提交前检查 `git diff --cached --stat` 与 `git diff --cached --check`；原始未跟踪文件（如两份 QA 渲染目录、修订稿 docx）不提交。

## 入口

- 执行顺序与当前任务：TODO.md 第 2 节；每任务的进入条件、验收标准与操作步骤：TODO.md 第 3 节。
- 常用命令模板与脚本接口约定：TODO.md 第 6 节。
- 独立审核提示词：TODO.md 第 6.5 节。

## 存储与运行

本地Python为`.venv_local/bin/python`。集群SSH别名`sblab`本轮实测可登录；项目根目录`/lenovofs1/home/jyma/PLM_benchmark/`。已有模型与embedding缓存沿原manifest登记使用；本轮统计及元数据纠错无需PLM前向。

本轮交付说明：[repair_summary.md](reports/repairs/20261001/repair_summary.md)。纠错锁校验：`.venv_local/bin/python scripts/verify_correction_lock.py`；原始结果和锁保留，修正结果在`results/repairs/20261001/`。

Phase6实施计划与后续验证补证草案：[Phase6_and_validation_TODO_20261002.md](deliverables/Phase6_and_validation_TODO_20261002.md)。本轮用户要求提供计划，P6/V任务保持未开始。20261001冻结材料中的Gate状态对应其冻结时点，实时治理状态以TODO与决策日志为准。

## 文献与结果阅读

PYP 的物理发现、当前模型结果及后续方案见[PYP 中文说明](reports/pyp_readable/20261002/PYP_readable_report.md)。配套精读卡分别解释 Tenboer 2014 的结构观测和 Konold 2020 的环境相关动力学。此次阅读补充没有运行新模型，也没有完成新的独立验证。

## 仓库同步与后续更新

用户指定汇总仓库为 [wyqmath/ESM-Latent-Topology](https://github.com/wyqmath/ESM-Latent-Topology)。该仓库的既有流程保留，当前 PhysInfoBench 快照放到其 `physinfobench/` 子目录。同步范围与后续缺口见[仓库更新计划](deliverables/Repository_update_plan_20261002.md)。当前本地科学状态仍以 TODO、决策日志和修订证据矩阵为准；仓库上传不会改变任务或证据等级。

上传前修订与验证记录：[代码审核修复记录](reports/code_review/20261002/CODE_REVIEW_REPAIR.md)。环境锁来自本地 Python 3.14.6 环境；集群 Slurm 脚本仍需配置实验室路径、模型缓存和工具目录，干净环境的完整安装验收尚未完成。
