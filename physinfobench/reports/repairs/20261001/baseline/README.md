# PhysInfoBench

诊断蛋白质基础模型中物理信息的可用性与预测失败来源，分离五类底层瓶颈：输入信息不足、表示不可读取、聚合折损、读取器容量受限、分布偏移与泛化失败。研究依据为用户指定的 `PLM_Proposal_FInal.docx`（路径与 SHA256 见 `configs/source_manifest.yaml`；工作区内 `PLM_Proposal0920_修订1.docx` 为未指定版本，不得作为研究依据）。

- **主执行记录**：[TODO.md](TODO.md)（Phase 0–6，全项目 49 个最小任务；2026-09-22 三层治理变更后 Phase 1 含 P1.01–P1.12；执行闭环、审核与提交规则见其第 1、6 节）。
- **当前状态**：见 TODO 第 2 节进度行；本文件由 P0.01 建立。
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

## 存储（P0.03 前为占位）

本机为 macOS（darwin 25.4.0，arm64）；远程 GPU、存储配额与模型名单由 P0.03 登记并经讨论确认，在此之前不写任何具体承诺。
