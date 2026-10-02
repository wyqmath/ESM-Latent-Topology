# 交付封存独立审核

审核时间：2026-10-01 23:00 Asia/Shanghai。审核者：bootstrap_repair。最终结论 **PASS**（0 blocker、0 major）。本轮仅执行只读核验和临时目录负例测试；没有修改锁定源码、结果、标签、划分或科学报告，没有构建实际最终交付包，没有提交。

## 最终锁与范围

执行 `.venv_local/bin/python scripts/verify_correction_lock.py`，最终555项全部PASS，errors为空。kind为posthoc_correction_integrity，gate_approval=false；该锁不表示预注册、新独立评价或G5批准。

最终锁SHA256：`5724e209711545b736ecdf1990588964286548055a033dee82fe8cd248a3c65e`。

核验源码SHA256：

- scripts/verify_correction_lock.py：`ec7e2ce4274bcfbe9ddd913adf83b0c596c2344adb5c5c0681579cfb7afcea61`
- scripts/seal_repair_delivery.py：`51e54761fae6287853ed7b7bf2c77b3e96957e769e45a4ceee994ac9a3d4979f`

全部555个锁定文件均属于当前打包候选。生成本审核文件前，候选共831个互不重复、确实存在的文件；实际打包还会收入本审核及最后的可变治理文字，最终成员数由交付QC记录。六份当前科学报告、实质修复报告和R5审核链已纳入锁；repair_summary及本封存审核通过最终ZIP manifest覆盖，避免自指锁。早期锁v1/v2只保留历史审核用途。

首轮封存范围的两处问题已经处理：实质修订报告及审核记录补入最终锁；表示缓存从打包候选和当前锁中排除，原本地缓存不删除。当前候选没有CIF、生产或烟测表示NPZ，仅保留两份名为frozen_reader_coefficients.npz的小读取器系数。两份JSON系数与相应NPZ逐元素完全一致，也核对了JSON内登记的NPZ来源SHA。

## 负例与ZIP实现核验

在独立临时目录复制全部555个锁定文件和锁，未篡改时PASS。只向复制的scripts/cluster_bootstrap.py追加测试字节后，校验FAIL且唯一错误为该文件hash_or_size_mismatch；恢复原字节再删除该临时文件，校验FAIL且唯一错误为missing。原工作区该文件及最终锁仍PASS。

另用临时两文件fixture运行实际package函数，ROOT/LOCK/成员选择及git返回值仅在该临时测试进程中替换。没有运行整个仓库的最终打包。实际ZIP经CRC、成员唯一性、读回SHA和外部ZIP SHA核对全部通过；manifest逐项覆盖全部payload及两份DELIVERY元数据，manifest自身作为校验清单不进行自哈希。再次向相同目标运行package抛出FileExistsError，拒绝覆盖。源码同时核验为：先核对纠错锁，再收集成员；ZIP关闭后testzip检查CRC、核对成员集合及全部读回SHA，最后写外部SHA和QC。

## 原始证据保全与科学报告

独立读取git show 18c89ea:path，逐字节比较historical_integrity_check.json登记的全部27份归档，全部一致；另比较108个当前原标签、划分、结果和两份原运行锁，全部与18c89ea字节相同，其SHA也与登记值相同。计数采用实际数组长度，并非只相信JSON内PASS字段。

已读repair_summary及当前六份科学报告：confirmation、generalization、claim_evidence_matrix、aggregation_diagnosis、tasks/P5.07、tasks/P5.08。报告中的确认148/66分量、保留108/59分量、曝光131/81、事后17/27、打结修订AUROC与区间，均与已独立核验的statistics/exposure产物一致。聚合750/325、2279/1952及72记录144,000抽样，与R5.05独立审核一致。类型89链/50分量、链与分量Macro-F1及选择依赖，与保存预测复算一致。外部99链/22,646残基的AUPRC0.476074、基率0.215346、BACC0.503304、MCC0.015449、池化0.360061，与17位概率及保存系数独立核验一致。

旧105及99混层历史、99同层纠错、最新守卫后置验证分别表述；没有将固定参数开发拟合重建写成零重拟合。报告保留CA坐标未观测标签与实验内在无序对应关系未核证的限制，当前科学证据等级保持D/X，没有通过统计修正升级历史曝光数据的独立评价资格。G5待确认、Phase6未开始，与当前进度一致。

## 包外依赖

报告和交付README明确说明：统计复算可用包内保存预测；P5.08完整读取器重放依赖集群生产缓存。原CIF、论文、PLM权重和生产/烟测表示缓存维护在登记的本地或集群路径，不随本包交付。完整曝光重建还依赖旧项目endpoint源表，其绝对路径与SHA在exposure/input_checksums.tsv。缓存、实际作业源码与输入SHA在p508/h33_exact_metrics/replay_provenance.json；原CIF核查源在对应身份审计manifest。当前打包范围与这些说明相符。

最终ZIP本身仍须由主代理在提交后运行package，并以其CRC/SHA/QC结果确认最终交付文件；本PASS覆盖封存代码、当前锁、报告一致性及临时fixture验证。
