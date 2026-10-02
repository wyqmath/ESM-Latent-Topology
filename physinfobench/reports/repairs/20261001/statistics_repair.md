# P5 打结统计修订

完成记录：2026-10-01T21:22:24+08:00；完整集合计算记录 UTC=2026-10-01T13:19:23.560701+00:00；事后子集计算记录 UTC=2026-10-01T13:19:16.660626+00:00。

本轮利用保存的预测修正统计，不运行新的 PLM 推断，不选择新层、聚合器、模型或超参数。原 confirmation/holdout 的 scores.tsv、metrics.json 和首次读取记录保持原样。修订版产物位于 `results/repairs/20261001/statistics/full_set_final/`；事后敏感性位于 `statistics/posthoc_sensitivity/`。

## 缺陷与修复

原 P5.02/P5.05 knot AUROC 回调将重复抽中的样本聚合到字典后只取每个键的首个分数，丢失重复次数。新工具 `scripts/cluster_bootstrap.py` 使用选择次数作为每条链的 sample_weight，与逐次重复链行完全等价。组件被抽中时，每条已评价成员链均携带同一组件选择次数。点估计仍为链加权 AUROC；置信区间的抽样单位改为完整绑定分量，附链单位区间供历史比较。

采用曝光审计重建的完整全局连通图，而非仅局部 group_id。缺失组件 ID、重复样本 ID、非有限分数或非法标签会报错。类别退化抽样保留 NaN 标记并计数，分位数仅由有效抽样计算；固定 B 次，不通过补抽改变条件分布。

协议 B=2000、95% percentile CI、种子 [13,42,2026] 全部报告。2026 沿用原报告种子作为展示锚，另外两种子保留，不按结果选择。无新的 p 值或 Holm 裁决。

## 完整集合结果

| 集合 | 链/阳性/阴性 | 全局分量 | AUROC | 原缺陷区间 | 修订分量 CI，seed=2026 | 修订链 CI，seed=2026 |
|---|---:|---:|---:|---|---|---|
| confirmation | 148/22/126 | 66 | 0.946970 | [0.903611, 0.993513] | [0.864895, 0.995423] | [0.866656, 0.994439] |
| final_holdout | 108/25/83 | 59 | 0.967711 | [0.942852, 0.993594] | [0.932105, 0.996418] | [0.926076, 0.995040] |

所有完整集合、两个抽样单位和三个种子的有效抽样均为 2000，退化为 0。确认集修订分量 CI 下界在三个种子下为 0.864346/0.869401/0.864895，均低于 0.90。保留集对应为 0.930799/0.935483/0.932105。不能继续使用原确认 CI 下界超过 0.90 的表述。保留集区间数值不改变历史曝光判定。

## 配对长度对照

原逐链长度对照预测没有归档，本轮严格按原 runner 的冻结 LogisticRegression(C=1,class_weight=balanced,solver=liblinear,max_iter=1000,random_state=2026) 从同一新版 development 的 750 条链（127 阳性/623 阴性）重建 log1p(length) 拟合。没有读取确认/保留标签进行拟合或调参。软件为 Python 3.14.6、sklearn 1.9.1、numpy 2.5.3。两个集合长度 AUROC 均匹配原报告六位精度，重建系数和所有输入哈希在 JSON 保存。重建不是原 comparator checkpoint 的字节级复原，因此配对结果明确标记为重建对照。

| 集合 | 长度 AUROC | PLM−长度 | 配对分量 CI，seed=2026 |
|---|---:|---:|---|
| confirmation | 0.575577 | 0.371392 | [0.175256, 0.549052] |
| final_holdout | 0.711084 | 0.256627 | [0.104111, 0.425624] |

两种方法用完全相同的组件抽样权重，避免用两个独立区间代替配对差值。以上完整集合比较仍受历史开发曝光限制。

## 排除历史曝光闭包后的事后敏感性

采用曝光审计的 posthoc_sensitivity_eligible 字段，仅对已保存的预测取子集。全部样本的 P5 结果已被读取，fresh_preregistered 为 0；这些结果不恢复新确认集或新保留集资格，也不产生新的独立泛化主张。

| 集合 | 链/阳性/阴性 | 分量 | AUROC | 分量 CI，seed=2026 | PLM−长度及配对 CI | 退化抽样 |
|---|---:|---:|---:|---|---|---:|
| confirmation | 17/4/13 | 14 | 0.980769 | [0.878788, 1.000000] | 0.230769 [0.000000, 0.666667] | 16/2000 |
| final_holdout | 27/8/19 | 18 | 0.986842 | [0.922078, 1.000000] | 0.078947 [-0.037500, 0.323232] | 0/2000 |

确认子集仅 4 条阳性，分量抽样在三个种子下有 20/20/16 次单类退化。保留子集 PLM 对长度的配对差值区间跨 0，确认子集下界为 0。小样本事后敏感性无法确证 PLM 优于长度对照。

## 实现、验证与复跑

新增 `scripts/cluster_bootstrap.py`、`scripts/recompute_p5_statistics.py` 和 `tests/test_cluster_bootstrap.py`。历史两 runner 的 knot 统计改用该工具，评估时必须显式提供 `--knot-component-map`，记录组件映射 SHA256，报告所有冻结种子与配对差值。历史 lock 不重写；校验实测仍拒绝已变更的代码哈希，首次读取守卫保留。修订通过独立新目录的统计脚本执行。

六项 unittest 通过：手算带并列分数 AUROC、重复样本改变 AUROC、整组件成员共同重复、同种子一致与配对同抽样、单类退化计数、缺失/重复 ID 报错。四个脚本 py_compile 通过；两个历史 runner 的 lock 校验按预期失败，未执行评估。

复跑完整集合：

```sh
.venv_local/bin/python scripts/recompute_p5_statistics.py \
  --component-map results/repairs/20261001/exposure/component_map.tsv \
  --output results/repairs/20261001/statistics/reproduction_full_set
```

复跑事后敏感性：

```sh
.venv_local/bin/python scripts/recompute_p5_statistics.py \
  --component-map results/repairs/20261001/exposure/component_map.tsv \
  --posthoc-exposure results/repairs/20261001/exposure/evaluated_sample_exposure.tsv \
  --output results/repairs/20261001/statistics/reproduction_posthoc
```

输出目录非空会拒绝覆盖。每个 seed 的全部 B 抽样存为 TSV，主 JSON 保存协议、软件、输入 SHA256、原始指标副本和修订结果。此项使用本地 CPU，无 GPU 或集群作业。

## 适用限制

分量抽样处理当前绑定图内相关性；max TM 单侧近邻和家族层缺口仍需沿原报告披露。结构缺失链未被此次统计修复补齐。区间条件于已训练读取器和当前数据，不包含开发选择不确定性；历史曝光不能通过重采样消除。后续证据裁决须结合 exposure 修订，不能仅依据完整集合区间升级。

## 集群独立环境复算

2026-10-01 22:08核验作业814880 COMPLETED（53秒，4CPU）。集群QOS拒绝CPU-only请求，按要求保留1GPU资源，代码未进行GPU计算。远程Python3.11.16/numpy2.4.6/sklearn1.9.1，13个传输输入逐SHA核验；两集合类型（全集与事后子集）共48,000次抽样以及配对差值与本地结果一致，摘要最大差为0，逐次差小于1e-12。远程输出与provenance位于results/repairs/20261001/cluster/。独立实现审核另由exposure_repair和claims_audit完成。
