# G4 里程碑：聚合保真度与条件信息干预（Phase 4 完成）

时间：2026-09-27 09:3x Asia/Shanghai 成文（P4.09 整改后 10:0x 更新）。P4.01–08 已提交，
P4.09 随本轮审核后提交；等待用户 G4 确认后进入 Phase 5。

## 1. Phase 4 任务清单

| 任务 | 内容 | 状态 | 审核 |
|---|---|---|---|
| P4.01 | 冻结聚合对比设计（configs/aggregation.yaml） | ✓ c8e2759 | 一审 4M+8m→二审 pass（0B/0M/4m 措辞随提交订正） |
| P4.02 | 逐残基与固定聚合比较（18 臂+族判定） | ✓ 1a8e29f | 合并审核三轮（3M+5m→conditional_pass→pass） |
| P4.03 | 参数受控可学习聚合（L-ATT 双臂×3 种子） | ✓ 1a8e29f | 合并审核三轮（1B+2M+4m→pass→pass） |
| P4.04 | 聚合损失证据（reports/aggregation_diagnosis.md） | ✓ 1a8e29f | 合并审核三轮（2M+4m→1 blocker→pass） |
| P4.05 | 条件特征与置换负对照冻结（configs/interventions.yaml） | ✓ 917a3e7 | 三轮审核（3M+8m→1M+2m→pass） |
| P4.06 | PYP 干预矩阵（1 对，RUN） | ✓ d97b484 | 合并审核两轮 pass |
| P4.07 | RNase A 干预矩阵（2 对，RUN_DEGRADED） | ✓ d97b484 | 合并审核两轮 pass |
| P4.08 | BPTI 干预矩阵（STOP 预注册执行） | ✓ d97b484 | 合并审核两轮 pass |
| P4.09 | 信息增益汇总+确认假设+G4 材料 | ✓ 本提交 | 一审 1B+2M+4m→整改→复审见 logs/reviews/P4.09.md |

轮次口径注："三轮"=完整审核轮次（一审→整改→复审/终核）；P4.06–08 的收口核验
未单独计轮（与 git 提交标题 two-round 表述一致）；P4.02–04 与 P4.09 为三轮。

## 2. 核心发现

### 聚合保真度（C-AG1/C-AG2；development）

- **聚合折损存在且算子依赖**：全局任务上末端向量显著大幅劣于均值池化
  （K-LAST −0.109 AUROC，sig99fam；层 29 复现）；均值池化≈窗口均值池化≈逐残基
  读取+归约（全部 ns）——**对全局终点，逐残基读取不比先池化后读取读出更多**。
- **残基终点天花板**：池化口径 0.9996 下三算子不可区分（"未检测到≠无损"）；
  per-protein 描述（n=4）显示广播塌缩（−0.413）而局部窗口近无损（−0.014）。
- **可学习聚合无恢复（C-AG2 refutes）**：单头注意力（82,048 参数）在 knots 上
  **显著更差**（−0.0241 sig99fam；逐种子 CI 补算全负），disorder 天花板下 ns。
  边界：本预算与架构内成立，不宣称机制。

### 条件信息干预（C-IN3；案例级）

- 三系统干预矩阵：PYP（1 对）/RNase A（2 对）四臂矩阵完成；BPTI STOP 预注册执行
  （有效独立配对=0）。
- **置换对照在案例级失效**（重拟合系数精确反号恢复预测）→ 按冻结解释触发 refutes
  （对照失效）分支；判定量降级为构造事实：**M 状态盲（同序列表示逐位相同）+
  F_only≡M+F**——条件主导的构造级支持。
- RNase 整系统=输入充分性对照语义（F 与标签同义），不作为模型恢复物理知识的证据；
  BPTI 登记为证据空缺（非阴性结果）。

### 确认假设（→ Phase 5，configs/confirmatory_hypotheses.yaml）

有限集 3 条（阈值确认前冻结、发现集曝光登记）：
1. H-FSL2-RANK：确认集 strict 阳性排序（pass=recall@1000≥0.5；fail<0.2 或 percentile>0.05）。
2. H-KNOT-PRESENCE：确认集 AUROC≥0.80（fail：点<0.65 且 CI 上界<0.80；完整判据见 yaml）。
3. H-DISORDER-RES：确认集池化 AUPRC≥0.90（fail<0.75）。
不登记：FS-REGION/FS-L1/KNOT-TYPE/条件三系统/聚合算子（理由见 yaml not_registered）。

## 3. 工程登记（如实）

- 聚合管线两阶段化（按臂断点续跑）与检查点链：三次失败作业（813129/813130/813154）
  如实登记于 P4.03；813123 主动终止重构；MIL 臂协议偏离重算（813180）。
- 集群审计：knots 750 manifest 行=584 唯一缓存键（重复序列共享）——无数据丢失；
  fasta 暂存清理属正常运维。
- 全部作业遵守 SLURM+GPU 纪律；收口后 `ps -u jyma` 无遗留计算进程（已核）。

## 4. 残留（不阻塞 G4）

| 残留 | 处置 |
|---|---|
| L3 人工全文复核（17 篇） | Phase 5 前置，用户付费墙 PDF 支持下启动 |
| SaProt/3B 前向验证 | 集群可用后可排（P2.04 名单内备选模型） |
| epoch 数不可恢复（P4.03 accounting 偏差） | 已如实登记；后续任务检查点包含 accounting |
| BPTI 功能数值终点 | 待 L3 全文复核后走差异清单另跑 |

## 5. G4 确认请求

请确认：Phase 4 完成（聚合保真度与条件干预结论如上；确认假设 3 条已冻结），
进入 Phase 5（确认集复核与最终泛化评估；P5.01 将把 confirmatory_hypotheses.yaml
转写为 confirmation_lock.yaml 并记录 first_read）。

全部 Phase Gate：G0 已确认；G1 已确认（有条件）；G2 已确认（有条件）；
G3 已放行（用户 2026-09-26 指令）；G4 **已放行**（见下确认注）。

> **G4 确认注（2026-09-27 13:0x Asia/Shanghai）**：用户以指令放行本门并授权推进
> Phase 5，原文『继续推进phase5，直到做完』。放行语义=确认 Phase 4 完成与本报告
> 全部结论，并授权 Phase 5 启动（P5.01 起）。登记：decisions.md 2026-09-27 13:0x
> 条目；TODO G4 里程碑行与头部 Gate 链。
