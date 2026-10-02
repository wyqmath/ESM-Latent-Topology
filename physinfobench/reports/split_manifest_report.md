# P2.02 划分 manifest 报告（冻结参数执行）

时间：2026-09-24 03:27；种子 2026；比例目标 70:15:15（组级）。

## 分配结果
- 组总数 69084；development 48372（70.0%）/ confirmation 10359（15.0%） / final_holdout 10353（15.0%）。
- 强制 development 的组 38 个（S1/S2 历史方法选择暴露 + 受控三系统案例组）。
- 绑定边：端点∈背景簇 192；matched_set 17（全部同集合断言通过）。

## 各任务可评价量（每集合）
{
 "bpti": {
  "development": 9
 },
 "disorder": {
  "development": 2380,
  "final_holdout": 510,
  "confirmation": 447
 },
 "knot": {
  "confirmation": 247,
  "development": 970,
  "final_holdout": 184
 },
 "fold_switch": {
  "final_holdout": 8,
  "development": 81,
  "confirmation": 7
 },
 "pyp": {
  "development": 6
 },
 "rnase_a": {
  "development": 9
 }
}

## knots type 任务按集合类构成
{
 "development": {
  "3_1": 75,
  "4_1": 7,
  "5_2": 2,
  "5_1": 1
 },
 "final_holdout": {
  "3_1": 11,
  "5_2": 1,
  "4_1": 1
 },
 "confirmation": {
  "3_1": 14
 }
}

## L2 背景（fs_l2_background_manifest.tsv.gz）
- 背景蛋白+FS 端点统一聚类；各集合簇数=（蛋白数另见 P2.02 报告更正：dev 343,682/conf 68,654/hold 71,850）={'final_holdout': 10201, 'development': 47646, 'confirmation': 10227}。

## 纪律与边界
- 组为原子；未拆组、未放宽门槛；S1/S2 暴露组强制 development（历史方法选择不重包装为未见）。
- S3 legacy/S4 构建使用=历史接触记录（用户 2026-09-24 限定 4），不剥夺 confirmation/holdout 资格；正式模型评价历史=0（S5）。
- knots 结构近邻边未纳入绑定（登记残留，P2.03 Foldseek 补算复核；补算后如触发跨集合合并须重跑划分并保留旧版本）。
- confirmation/final_holdout 标签读取纪律：first_read 登记制（P5.01/P5.02）；本任务只生成 ID 映射，未读取任何标签进方法。
