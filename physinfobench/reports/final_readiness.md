# 最终评估就绪报告（P5.04）

任务：冻结模型、数据、超参数、特征、主要对比、指标公式和执行版本，审计保留集曝光。
时间：2026-09-27 14:38 开始；14:54 final_lock 冻结定稿（真实时刻）。
前置：确认阶段闭环（P5.01–P5.03：lock 441416a → 一次性复核 015aee3 → 三项裁决
e3e011c，均无方法修改）。

## 交付物

- **configs/final_lock.yaml**（locked_2026-09-27_p504）：
  - 门禁 gate.user_authorization_recorded（当前为空=禁止运行；用户明确授权后回填）；
  - first_read 注册（06:54Z，早于任何保留标签读取；运行器守卫+强制落 first_read_record）；
  - 保留集构成钉版：knots 可评价 25 阳/83 阴=108（3pvm_B 无序列，双口径披露）、
    disorder 513=494+5+14（51,134 分母域残基）、fs 7 对 n_strict=0（不运行、无
    side-table——gate.fsl2_holdout_decision）；
  - 训练策略事先明确：读出仅 dev 拟合（750 / 2,279 蛋白），**不合并确认集入训练**；
  - 环境锁（python 3.11.16 / torch 2.6.0+cu124 / sklearn 1.9.1 / numpy 2.4.6，venv）；
  - 执行矩阵/预期输出/资源预算（1 GPU 作业 ≤1h）/失败重跑规则/输出 schema；
  - 完整性钉版 25 文件（含新增 build_p504_holdout_inputs.py、run_p505_holdout.py、
    sbatch_p505.sh 三脚本哈希）。
- scripts/build_p504_holdout_inputs.py（已试运行：108=25+83、缺失恰 3pvm_B、
  494+5+14=513、fs 7 对 n_strict=0 全部断言通过）。
- scripts/run_p505_holdout.py（保留集一次性评估器；gate 强校验缺失即 die；
  first_read 守卫；first_read_record.json 由运行器强制写入——P5.02 缺陷设计级修复）。
- scripts/sbatch_p505.sh。

## 就绪核查（验收条件逐条）

1. **确认阶段完成**：P5.03 三项裁决记录于 decisions.md 14:38 条目；确认批关闭
   （exposure_log 状态行）；无未闭合回路。
2. **Git 版本与配置校验值一致**：本地 verify（final_lock 25 文件）实测通过；提交后
   rsync 集群复验（同 confirmation 流程）；工作区无未提交改动混入（提交前 git status
   断言 clean）。
3. **保留集曝光审计**：结果级读取=0（exposure_log 17 数据行+表头逐行核验，无 holdout 结果读取行）；保留集历史接触仅=划分生成（标签无关，2026-09-24 已登记）+ 聚合构成
   统计（split_qc 26/83、type 构成，已登记）。
4. **仅冻结并确认后解锁最终测试**：gate 字段为空时 run_p505_holdout.py 首步即 die；
   授权回填方式=decisions.md 条目引用（用户明确指令），再由收口提交写入 final_lock
   （append-only change_log）。

## 最终执行矩阵（P5.05 待授权）

| 运行 | 内容 | 对照 |
|---|---|---|
| HOLD-KNOT-PRESENCE-P505-v1 | knots holdout 108 评分，AUROC+CI | 置换×100 + 长度负对照 |
| HOLD-DISORDER-RES-P505-v1 | disorder holdout 494 蛋白池化 AUPRC+CI | 蛋白内置换×100 + 基率 + 双类计数 |
| HOLD-FSL2-RANK | 不运行（n_strict=0 预注册分支） | — |

判读纪律：与确认阶段方向不一致时按泛化失败/不确定性报告，不回传调参；disorder 的
双类蛋白计数为判读前置量（P5.03 退化先例）。

## 门禁请求（提交本报告后向用户提出）

请求授权：final_holdout 解锁，允许 P5.05 按上述冻结矩阵执行**一次性**保留集评估
（访问次数=1；原始预测与日志全量保存；评估后无论结果如何不回传调参）。
授权方式：任意明确指令即可（将登记 decisions.md 并回填 final_lock.gate）。

## 完成记录

- 审核记录：一审 **pass（0B+0M+3m）**（agent_f20677eb；负面测试实测 gate 拒绝、
  哈希三重验证、计数三方复算一致）；3 MINOR 随提交订正（见 logs/reviews/P5.04.md）。
- Git 提交：本提交（哈希见 TODO P5.04 行）。
- 状态：P5.04 完成，**final_holdout 解锁门禁待用户明确授权**；授权回填方式=
  decisions.md 登记 + final_lock.gate 回填（append-only change_log）后 P5.05 方可运行。
