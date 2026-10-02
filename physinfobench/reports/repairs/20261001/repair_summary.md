# PhysInfoBench 本轮修复与结果交付

封存准备时间：2026-10-01 22:55 Asia/Shanghai；开始：2026-10-01 21:08。原始基线：18c89ea。本轮R5.00–R5.05逐项修复、独立审核、单独提交；实际提交通过 `git log --grep=R5` 查询，交付包元数据另记最终HEAD。用户授权修复和并行执行；G5尚待确认，Phase6尚未开始。

已完成历史开发曝光与绑定闭包审计、bootstrap重复次数修复、完整分量统计重算、外部蛋白身份映射、表示层错配修复、聚合比较纠错和证据报告修订。当前数据支持开发性诊断与事后补充分析；独立确认/保留泛化结论的证据不足已经写入矩阵。

## 修复结果

| 项目 | 修复结果 | 对结论的影响 |
|---|---|---|
| 打结历史曝光 | 确认131/148、保留81/108落入历史开发绑定闭包；全图68,875分量 | 原C/F独立泛化支持撤回，保留D级描述 |
| 打结完整集合统计 | 确认AUROC0.946970，95%分量CI[0.864895,0.995423]；保留0.967711，[0.932105,0.996418] | 点值保留，原重复抽样实现导致的区间停止使用；统计修正不能消除历史曝光 |
| 打结事后敏感性 | 剩余17/27链的PLM−长度差为0.230769，[0,0.666667]及0.078947，[−0.0375,0.323232] | 全部样本已评价，未确证优于长度，不成为新确认集 |
| 未来划分守卫 | 全分量覆盖、真实成员ID和历史使用均检查；当前池有效未曝光打结双类评价集为0 | 历史开发样本及已消费测试不能重新变成未见测试 |
| 外部结构补充集 | 从239候选复原历史105链；5无身份+1多身份隔离，保留99链/22,646残基 | 精确链→entity→accession；36候选DisProt重叠此前已由同源步骤排除 |
| 外部表示层纠错 | 原开发H11→外部H33；本轮同层H33→H33：逐链AUPRC0.476074，基率0.215346，BACC0.503304，MCC0.015449；池化AUPRC0.360061 | 排名信号高于基率；固定阈值性能接近机会水平；CA坐标未观测仍为代理标签，X级事后补充 |
| 聚合对比统计 | 750打结链/325分量，2279无序蛋白/1952分量；72记录、144,000抽样 | 末端向量差−0.1090、三训练种子均分注意力差−0.0241的族区间均为负；MIL/窗口无明确差值，未做等效性检验 |
| 打结类型表述 | 链Macro-F1=0.397、分量0.394；同轮LCO用于选择与报告；固定预测置换24/101 | 高3_1召回不等于有效多类区分，绑定分量数不等于生物家族数 |

区间展示锚为原冻结bootstrap种子2026，B=2000；13/42/2026三种子结果全量保存。聚合族区间沿原3/3/2主比较校正；训练种子与bootstrap种子分列。统计均条件于保存预测和既有选择。

## 日志与结果入口

- 当前科学裁决：reports/claim_evidence_matrix.md；确认与保留分别见reports/confirmation.md、generalization.md。
- 各修复报告与首审失败/最终PASS：reports/repairs/20261001/；逐项审核logs/reviews/R5.*.md。
- 曝光与可复算分量：results/repairs/20261001/exposure/；其中current_split_future_guard.json为早期仅标签候选检查，不作为全局划分通过证明。全局审计及独立守卫验证见exposure独立报告。
- 打结修正、配对长度、逐次bootstrap：results/repairs/20261001/statistics/。
- 聚合修正与原保存OOF：results/repairs/20261001/aggregation/。
- 外部身份、239排除复原、旧混层99及当前同层99：results/repairs/20261001/p508/。h33_exact_metrics/为当前纠错结果；exact_metrics/为历史混层重放。summary.json为早期身份阶段摘要，不代替h33_exact_metrics/metrics_exact.json。
- 集群作业日志及统计独立环境复现：results/repairs/20261001/cluster/；P5.08作业日志在p508/slurm_*.out。
- 旧文件保全：27份归档与18c89ea原件逐字节相同；108个原标签/划分/结果及确认/保留锁全部未改。结果见historical_integrity_check.json。旧报告正文与当前科学裁决分开保存。
- 单独纠错锁：configs/p5_correction_lock_20261001.yaml；用于登记已完成纠错的输入、代码和输出；G5仍待确认。

## 核验与计算

23项有意义测试覆盖抽样重复次数、配对抽样、绑定完整性、ID真实性、历史曝光、accession连接和按元数据取层。打结48,000次抽样在集群独立Python/numpy环境复算，摘要差为0。聚合独立审核核对576次sklearn抽样、全部144,000次记录分位数及退化计数。外部保存系数在99个缓存直接复算22,646分数，最大差4.44e-16；每链/池化/100次固定预测置换均一致。首审发现的问题及修正后的版本均保留。

集群sblab实测可用，四个作业分别为814879失败2秒（拟合前输入索引检查）、814880统计复现53秒、814881历史混层重放61秒、814885同层纠错47秒。任务使用4或8CPU；该集群QOS要求保留1GPU资源，代码未使用GPU执行PLM前向，既有表示缓存被复用。CPU资源与64GB请求属于运行分配，不表示实际占用64GB。

## 可执行复跑

本地环境`.venv_local/bin/python`：Python3.14.6、numpy2.5.3、scikit-learn1.9.1；独立集群环境Python3.11.16、numpy2.4.6、scikit-learn1.9.1。先核对纠错锁，再在新目录复算，旧锁会拒绝已改源码。

```sh
.venv_local/bin/python scripts/verify_correction_lock.py
.venv_local/bin/python -m unittest discover -s tests -v
.venv_local/bin/python scripts/recompute_p5_statistics.py --component-map results/repairs/20261001/exposure/component_map.tsv --output results/repairs/20261001/reproduction_full
.venv_local/bin/python scripts/recompute_p5_statistics.py --component-map results/repairs/20261001/exposure/component_map.tsv --posthoc-exposure results/repairs/20261001/exposure/evaluated_sample_exposure.tsv --output results/repairs/20261001/reproduction_posthoc
.venv_local/bin/python scripts/recompute_aggregation_statistics.py --output results/repairs/20261001/reproduction_aggregation
```

外部精确指标可由包内17位分数复算；完整读取器重放依赖集群已有缓存。实际源码、输入manifest、缓存哈希、层审计和系数在p508/h33_exact_metrics/replay_provenance.json等文件登记。NPZ与可入Git的JSON系数逐元素精确相同，交付包包含二者。原CIF、论文和embedding缓存沿原登记路径维护，不打入日志结果包。完整历史曝光审计还依赖旧项目endpoint源表，路径与SHA在exposure/input_checksums.tsv。

## 本轮后的工作

修复后G5材料可供确认。恢复独立泛化评价需要另建尚未曝光的数据、冻结新批次后再评价；现有池重划分无法增加未曝光身份。FS仍缺独立strict阳性，17个L3候选全文资格尚待复核；无序任务仍需要双类实验标签。SaProt/3B及Phase6写作、图表、正式复现发布属于后续任务，本轮没有将它们标为完成。
