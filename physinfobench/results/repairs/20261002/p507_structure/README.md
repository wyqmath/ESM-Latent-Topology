# P5.07 双向结构确认与开发探针修订

本目录保存 P5.07 修订计算的可发布结果。集群原始 mmCIF/PDB 文件、模型权重及完整表示缓存未打包；输入来源及路径按项目数据清单登记。

- `usalign_edges_v2.tsv`：3,010 条新链—已有链配对的双向归一化 TM-score、min TM 与进程状态。每行均成功解析；2,638 条 min TM≥0.6。
- `newnew_edges_v2.tsv`：7,962 条新链—新链候选的同格式确认记录；7,226 条 min TM≥0.6。
- `foldseek_hits.tsv`、`nn_hits.tsv`、`seq_clusters_rep.tsv`：候选边和序列簇关系。最终宇宙构建器按候选全集核对确认表的配对覆盖，缺失、失败、重复和分数不一致均会停止。
- `p507_final_universe_20261002.json`：基于冻结类型表、旧打结开发样本、序列簇和双向结构边构建的版本化开发宇宙。157 个候选节点中 31 个触及 confirmation/final-holdout 连通分量而剔除；保留 126 条链、53 个分量。
- `selection_grid.tsv`、`lco_predictions.tsv`、`metrics.json`：开发探针参数选择、预测与指标。选中 hidden 33 / C=1；Macro-F1=0.4346、balanced accuracy=0.5142。参数选择与报告使用同一非嵌套 LCO，指标只作开发性描述。4_1 有 4 个分量、5_2 有 3 个分量，结论按案例级解释。
- `QC.json`：文件摘要哈希、配对数、通过数和本地/集群构建一致性检查。

Slurm 作业：814986（新—新 US-align）、814987（P5.07 fp32 表示）、814989（新—旧结构表验证）、814993（宇宙重建和开发探针）；均以退出码 0 完成。分析脚本与冻结口径见 `scripts/p507_confirm_newnew.py`、`scripts/p507_isolate_cluster.py`、`scripts/p507_build_final_universe.py`、`scripts/p507_type_probe_20261002.py`、`configs/p507_type_probe_design.yaml`。
