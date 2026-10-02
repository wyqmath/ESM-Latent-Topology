# 保留集泛化评估报告（P5.05）

执行：2026-09-30 19:41–19:52（作业 814839，**一次性**，访问次数=1）。
授权：用户 2026-09-30 指令式（decisions.md 19:4x 条目；G3/G4 先例语义）。
锁定依据：configs/final_lock.yaml（de9cc04 冻结 + gate 回填 change_log；job 内
25 文件 sha256 校验通过）。训练策略：仅 development 拟合（不合并确认集）。
first_read：2026-09-30 19:50（两 run 同批；first_read_record.json 运行器强制写入）。

## 1. 一次性执行总览

| 运行 | 结果（按冻结判据机械判定 + 判读） |
|---|---|
| HOLD-KNOT-PRESENCE-P505-v1 | **AUROC 0.9677，95% CI [0.9429, 0.9936]**——超 strong_pass 线（≥0.90） |
| HOLD-DISORDER-RES-P505-v1 | 机械 0.9999≈基率 0.9989；**双类蛋白=1/494**——与确认集同构的退化（预注册 degeneracy_note 命中） |
| HOLD-FSL2-RANK | 不运行（final_lock 钉版 n_strict=0；无 side-table） |

## 2. H-KNOT-PRESENCE（主张级：最终泛化证据）

- 保留集 108（25 阳/83 阴；knot:3pvm_B 无序列双口径披露；声明 26/83）。
- AUROC **0.9677**，蛋白单位 bootstrap B=2000 CI **[0.9429, 0.9936]**（CI 下界即超 0.90）。
- 对照组（全部预注册）：标签置换零分布 mean 0.499（max 0.729）；链长单特征负对照
  AUROC **0.711**——链长在保留集解释力升高（vs 确认集 0.576）但仍远低于读出，
  信号非长度可解释。
- 三级证据链齐备：development 0.929 → confirmation 0.947 → **final_holdout 0.968**。
  方向一致、幅度稳定——**C-GE1/T-KNOT-PRESENCE 获最终泛化级支持**（本项目唯一一条
  走完 dev→确认→保留全链的主张）。

## 3. H-DISORDER-RES（判读=不可判读，与确认集同构）

- 保留集 494 蛋白 / 51,134 残基 / 正例率 0.9989；**双类蛋白=1**（DP01351，其
  per-protein AUPRC=0.984 为单例描述，无统计意义）。
- 池化 0.9999 vs 蛋白内置换零分布 mean 0.9994（max 0.9996）——与确认集相同：
  观测贴着零分布，基率伪影。
- 判读：DisProt 源的确认/保留子集**结构性缺乏类别对比**（双类 0→1），任务在这两个
  集上不可判读——P5.03 裁决在保留集复现。外部 X-ray 对比集（P5.08，105 蛋白
  AUPRC=2.0×基率）为该支腿提供独立模态的补充证据。
- 无回传调参：任何修改须走差异清单+新数据（final_lock failure_rerun_rule）。

## 4. 覆盖率、失败与不确定性

- 抽取：knots 108/108、disorder 494/494 零缺失（die-on-missing 全程有效）；
- bootstrap 有效抽取：knots 2000/2000；disorder 2000/2000（无退化丢弃）；
- 无中途选择/赢家重跑（one_shot_note 落盘 holdout_qc.json）；
- 原始预测全量：scores.tsv（108 行）、residue_scores.tsv.gz（51,134 行）。

## 5. 与确认阶段方向比较（TODO P5.05 第 5 条）

knots 方向一致且更强；disorder 同构退化（预注册预期内）；无泛化失败信号。
新增分析：无（本报告不产生任何事后探索；fs holdout 7 对无 strict——证据缺口如实
保留，联动 L3 全文复核专项）。

## 6. 交付物与账目

- results/holdout/{HOLD-KNOT-PRESENCE-P505-v1, HOLD-DISORDER-RES-P505-v1}/
  （metrics.json + 原始预测全量）；holdout_qc.json；first_read_record.json
- logs/slurm_phase5/p505_holdout_814839.out（+814837 校验拦截留痕）
- logs/exposure_log.tsv 结果级行补登（真实时间戳 19:50）
- 报告：本文件 + reports/tasks/P5.05.md

## 7. 边界

- final_holdout 已消费（访问=1，不可重测同锁）；后续任何改动走 final_lock 变更治理；
- knots 证据边界：条件无关全局属性，不外推打结类型；
- disorder："DisProt 确认/保留子集不可判读"是数据性质结论，非模型阴性。
