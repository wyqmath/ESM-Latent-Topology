# FS-L2 FP32 修订复算（2026-10-02）

本目录保存上传前代码审核后对 T-FS-L2-PU-RANK 的精度复核产物。模型为 ESM-2 650M，revision `08e4846e537177426273712802403f7ba8261b6c`，CUDA FP32 前向。此次复算只验证冻结的 development 流程，不读取 confirmation 或 final-holdout 标签，也不构成独立泛化验证。

## 结果

- stage-A 从冻结层集 `[5, 11, 17, 23, 29, 33]` 选择 hidden 29、`C=0.1`。
- 当前 universe 为 342,014（342,004 个背景 + 10 个 strict 阳性）；阳性一基排名为 `1, 2, 3, 4, 5, 6, 7, 8, 28, 29`。
- mean positive rank / universe size 为 `2.7e-05`；recall@10/100/1000/10000 为 `0.8/1.0/1.0/1.0`。
- 结果与当前划分下已保存的历史 stage-B 点估计逐项一致。
- 旧划分记录的 universe 343,692 与 percentile 0.000143 属于旧划分版本，不与本次 current-split 指标合并。

## 输入覆盖

当前划分需要 47,595 个 stage-B 背景代表。基础 FASTA 有 47,604 条，覆盖当前代表中的 45,133 条；其中 2,471 条是旧划分多出的代表。Delta FASTA 的 2,462 条补齐了当前集合。两份 FASTA 无交集，并集恰好覆盖当前全部 47,595 个代表。核查结果见 `l2_stage_b_input_union.json`。

作业 814994 完成基础代表、stage-A 和端点提取后，在检测到缺失 delta 时停止，没有写出 stage-B 指标。作业 814995 只提取缺失的 delta，随后运行 stage-B。两批表示均在 FP32 下生成；每批逐样本 manifest 见 `extract_manifests/`，汇总版本与哈希见 `extraction_provenance.json`。

## 文件

- `T-FS-L2-PU-RANK.stageA.selection.tsv`、`*.stageA.selected.json`：完整 stage-A 选择记录。
- `T-FS-L2-PU-RANK.stageB.metrics.json`：官方排序指标与当前宇宙大小。
- `T-FS-L2-PU-RANK.stageB.scores.tsv.gz`：传播到全宇宙的分数表（342,004 个背景条目）。
- `T-FS-L2-PU-RANK.stageB.propagation_qc.json`：传播覆盖与校验信息。
- `l2_stage_b_input_union.json`：基础与 delta FASTA 对当前代表集合的覆盖核验。
- `extract_manifests/`：stage-A、stage-B 基础/增量和 strict 端点的逐样本提取记录。
- `extraction_provenance.json`：模型 revision、精度、设备、manifest 哈希及历史值比较。

对应 Slurm 日志在 `reports/code_review/20261002/cluster/`。
