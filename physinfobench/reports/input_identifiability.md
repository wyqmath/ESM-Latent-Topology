# 输入可识别性核查（P3.01）

时间：2026-09-24 16:59。脚本=scripts/check_input_identifiability.py（断言全过才产出；qc=input_identifiability_qc.json）。
问题：各任务输入（纯序列 M／条件 F／残基级）能否区分其真值？同序列跨条件等价组（同输入不同真值）逐组列出；
此类目标纯序列下不可识别——只能作条件任务（需 F）、潜能/区域语义任务或**输入不足对照（BN1）**，
**不得据此判编码器失败**（任务验收条件原文）。全程只读冻结产物，未按样本读取/使用 confirmation/final_holdout
标签值（聚合计数覆盖全体冻结表行，含 conf/holdout 行的计数贡献）。

## 1. 有效独立单位（manifest 复算）

注：knot 区 manifest 1,401 行含 381 行 presence_mask=0 隔离样本（不入评价分母）；presence 可用=1,020（阳性 188+阴性 832）；type 可用=112。

| 任务区 | 样本 | 独立组（manifest group） | 三集合分布 |
|---|---|---|---|
| bpti | 9 | 1 | dev 9 / conf 0 / hold 0 |
| disorder | 3337 | 3334 | dev 2380 / conf 447 / hold 510 |
| fold_switch | 96 | 89 | dev 81 / conf 7 / hold 8 |
| knot | 1401 | 1039 | dev 970 / conf 247 / hold 184 |
| pyp | 6 | 1 | dev 6 / conf 0 / hold 0 |
| rnase_a | 9 | 1 | dev 9 / conf 0 / hold 0 |

## 2. 同序列跨条件等价组（纯序列不可识别的证据组）

| 组 | 输入等价 | 真值差异 | 任务影响 |
|---|---|---|---|
| FS 端点同序列对（96 对全集中） | 19、23、45、49、67、78、87、95（共 8 对，两端观测序列 sha 相同） | 同一序列对应两个状态结构 | **strict 范围内仅 porter_87**（本脚本断言，与 P1.20 口径一致）；8 对中其余 7 对均为 pending/extension（不在任何监督分母）。porter_87 状态判别臂不可识别→作 BN1 信息不足对照；区域/潜能臂不受影响（区域标签为 pair 级） |
| PYP | WT P16113 125aa（4WL9 与 4WLA 同序列；identity_check_pending 如实登记） | dark vs light_intermediate | 条件状态任务；条件特征 pending_fulltext→阻塞 |
| RNase A | P61823 同一序列（5 个有标注组装样本） | monomer×2 / C-swap dimer / N-swap dimer / trimer | 纯序列不可识别；有效独立组=1→AUROC 主张不成立；保留为 C-IN3 输入充分性对照 |
| BPTI | 同变体序列（氧化还原不改变残基序列）；ox_and_red 变体 4 个（BPTI-SD-5-55、BPTI-SD-14-38、BPTI-SD-30-51、BPTI-SD-5-51） | oxidized vs reduced（同一序列） | 区分性条件=氧化还原处理本身≈标签同义→无有效 F；保留为 BN1 信息不足对照 |

跨 pair 同序列端点组=0（无跨任务真值冲突）。FS 端点序列来源=旧项目 endpoint summary（192 端点）。

## 3. 逐任务判定

| task_id | 输入 | 目标 | 可识别性判定 | 处置 | 关键注记 |
|---|---|---|---|---|---|
| T-FS-GLOBAL | — | 伞任务（2026-09-22 起细化为 L1/L2/L3） | 不单独设探针 | **不适用（引用兼容保留）** | 由三层子任务承载 |
| T-FS-L1-PAIRED | M_residue/M_pooled（纯序列）+ pair 结构 | 阳性内部机制诊断（多态潜能读出+区域定位） | 部分可识别：区域/潜能臂可识别；状态判别臂仅 porter_87 同序列双态不可识别 | **白名单（去掉状态判别臂主张）** | porter_87=strict 内唯一同观测序列双端对（本脚本断言）——作 BN1 信息不足对照，不判编码器失败 |
| T-FS-L2-PU-RANK | M_pooled（纯序列） | 已知阳性在固定未标注宇宙中的富集（潜能，PU observed 语义） | 可识别（潜能语义） | **白名单** | 标签=experimental potential，非状态判别；措辞按 P1.12 冻结边界 |
| T-FS-L3-MATCHED | M_pooled + 匹配设计 | 病例 vs 操作性对照区分 | 设计可识别；确证性证据未闭合 | **阻塞（非可识别性原因）** | 17 对照全部 pending_manual_fulltext（G2 适用范围 2）；复核通过前不入确证主分析 |
| T-FS-REGION | M_residue（纯序列） | 转换核心区残基定位（潜能/区域，pair 级标签） | 可识别（区域潜能；porter_87 同序列两端共享同一区域标签，无冲突） | **白名单** | 主分析口径=双满足（fine∧usable）3 对 6 行（porter_20/61/62，P1.15 冻结）；fine_only 6 对 10 行=敏感性层；fine 9 对仅为坐标可靠性上限，不得称 9 对可靠标签 |
| T-KNOT-PRESENCE | M_pooled（纯序列） | 整链打结存在性（条件无关全局属性） | 可识别（无同输入异真值通道） | **白名单（提取前须补序列连接+同序列异标签断言）** | knots.tsv 无序列列——P3.03 提取前从 PDB/UniProt 取序列并加 same-sha×label 冲突断言 |
| T-KNOT-TYPE | M_pooled（纯序列） | 结拓扑类型多分类（条件无关） | 可识别 | **白名单（dev 层）** | 确认集仅 3_1（14 行）、5_1 仅 dev——G2-REV 复算口径；类缺失按协议 NA |
| T-DISORDER-RES | M_residue（纯序列） | 残基级有序/无序（潜能） | 可识别（歧义位由 P1.05 掩码出分母） | **白名单** | 掩码表 4,981 区段：state1=4870/state0=7/NA=104；13 条目含 X/U/Z 残基身份歧义（P1.09 隔离，提取时按掩码处理） |
| T-PYP-STATE | M + F（光照/延迟/环境） | 同序列光状态（条件状态） | 纯序列不可识别（同 WT P16113 序列 dark vs light_intermediate）；条件特征 pending_fulltext | **阻塞（条件任务证据未闭合；identity_check_pending）** | 4WL9/4WLA 同序列双态=BN1 对照可用；F 到位后转条件任务 |
| T-RNASE-ASSEMBLY | M + F（化学计量/组装环境） | 同序列组装状态（条件状态） | 纯序列不可识别（同 P61823 单体/C-swap/N-swap/trimer 四态）；有效独立组=1；条件定量字段 pending_fulltext | **阻塞为性能任务；保留为输入充分性对照（C-IN3 降级语义）** | n 独立组=1 不支持 AUROC 主张；条件≈标签同义风险已在 claims 预登记 |
| T-BPTI-REDOX | M + F（氧化还原处理） | 同变体氧化/还原状态（条件状态；功能终点） | 纯序列不可识别（同变体 ox/red 同序列；ox_and_red 变体 4 个）；区分性条件=标签同义（redox 处理本身），无有效 F | **阻塞为条件任务；保留为 BN1 信息不足对照** | 变体序列本地缺失（可由 WT P00974+Cys→Ala 位点确定性构造，构造后需核对）；终点=trypsin 结合（功能，不得换名） |

## 4. P3.02/P3.03 可执行白名单与阻塞项

**白名单（探针配置与运行可直接纳入）**：T-KNOT-PRESENCE、T-KNOT-TYPE（dev 层为主，类缺失按协议 NA）、
T-DISORDER-RES、T-FS-REGION（主分析口径=双满足（fine∧usable）3 对 6 行 porter_20/61/62；fine_only 6 对 10 行=敏感性层）、
T-FS-L2-PU-RANK（dev 宇宙）、T-FS-L1-PAIRED（区域/潜能臂；无状态判别臂主张）。

**阻塞项（非可识别性原因即注明）**：
1. T-FS-L3-MATCHED——17 对照逐例人工全文复核未完成（G2 适用范围 2），复核通过前不入确证主分析。
2. T-PYP-STATE——条件特征 pending_fulltext + 序列同一性 identity_check_pending；F 到位后转条件任务，当前 4WL9/4WLA 可作 BN1 对照。
3. T-RNASE-ASSEMBLY——有效独立组=1+条件定量字段 pending_fulltext；按 claims 预登记降级为输入充分性对照。
4. T-BPTI-REDOX——F≡标签同义、变体序列本地缺失（可由 WT+disulfide_pair 位点确定性构造，构造后核对）、终点=功能；保留为 BN1 对照。

**提取期统一前置（P3.03 preflight）**：knots 链序列本地缺失（knots.tsv 无序列列）——提取前须从 PDB/UniProt 取序列，
并对全任务统一执行 same-sha×不同标签 断言（本报告对 FS/PYP/RNase/BPTI 已做，knots/disorder 在序列连接时补做）。

## 5. 与 BN1（输入信息不足）的映射

- porter_87 同序列双端（strict 内唯一）→ L1 状态判别臂的天然信息不足对照。
- BPTI ox/red（同变体同序列，F≡标签）→ 条件贡献臂的信息不足对照。
- RNase 四态（同序列，独立组=1）→ 组装条件臂的信息不足对照。
- PYP dark/light（同序列，条件 pending）→ 条件特征到位前同上。
以上对照的**预期失败**（M-only 臂区分不开）是设计内结果，用于证明条件字段必要性，不构成对编码器能力的否定。

## 6. 措辞边界

1. 『纯序列下不可识别』仅指输入信息不等价（同输入多真值），不指模型/编码器失败。
2. FS 的 L2 目标=『已知阳性在固定背景中的富集（潜能）』，不得写成『状态判别』；L1 结论限 strict 阳性内部（claims claim_boundary）。
3. 本页全部『阻塞』均为证据/数据可得性阻塞（复核、全文、序列构造），无一是以可识别性为由叫停的可识别任务。
