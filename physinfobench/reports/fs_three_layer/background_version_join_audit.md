# 背景版本跨源连接核查（P1.21；G1 材料修订）

时间：2026-09-23 21:51–21:58 Asia/Shanghai。指令来源：用户 2026-09-23 21:05 第 3 项（核查接受背景版本差异所需的跨源连接；证据足够时再建议接受，不预写"影响仅限覆盖分母"）。产物：background_version_join_audit.tsv（35 键×15 列）+ qc；脚本 scripts/audit_background_joins.py（只读，exit 0；断言：strict=10、候选=25 且全在 B 表）。

## 1. 核查范围与连接定义

键集=10 strict 阳性（uniprot_a）+25 个 L3 候选（pna_evidence_review 全集，含主选/备选/2 标记）。四源版本：UniProt FASTA 2026_03（A/B 覆盖分母与 PN sp_length 源）｜SIFTS 2026-09-13（PDB 37.26/UniProt 2026.04：映射段+observed 段+uniprot_pdb）｜SIFTS 2026-09-20（PDB 38.26/UniProt 2026.03：pfam/cath/scop2+pubmed）｜wwPDB 2026-09-12（entries.idx+entry_type）。

六类连接：J1 长度（fasta vs REST 当前 vs SIFTS 隐含 SP 上界）；J2 家族存在性；J3 PubMed 存在性；J4 wwPDB 存在性；J5 全局 PDB 集偏差；J6 覆盖复算（fasta 分母+区间并集同口径）。

## 2. 结果

**一致项（无错配）**
- 候选长度 25/25 一致（fasta 2026_03=SIFTS sp_length 同源）；无一键出现 SIFTS SP_BEG/SP_END 超出 fasta 长度（无 2026.04→2026.03 序列伸长信号）。
- 阳性长度 8/10 一致（fasta=REST 当前；序列逐字节一致已在 P1.18 复核）。
- 覆盖复算 25/25 候选与 B 表存储值逐位一致（mapped/observed coverage max）。
- wwPDB 存在性：35 键的全部 09-13 PDB 均在 entries.idx/entry_type（0 缺失）；entry_type 无 computational 混入。

**发现（三类，均登记）**
1. **2/10 阳性无 fasta 锚（TrEMBL-only）**：Q8E473（porter_72，1310 aa）、Q9YA14（porter_61，161 aa）不在 Swiss-Prot 2026_03 fasta 内——canonical 序列与长度仅锚定于 UniProt REST 当前（P1.13/P1.09 审计 fasta，2026-09-23 取）。影响：这两对的任何 fasta 锚定连接（长度归一、A 宇宙同源比对）无法用冻结源执行，须以"REST 锚+版本锁定例外"登记（P1.10 的 not_in_sprot 排除通道已预告此类键存在于 FS 池）。
2. **家族层（09-20）注记缺口**：阳性侧 5 个 PDB 无任何家族行——B9W5G6（porter_68）的 6k2g/9gkl/9gkp（17 个中 3 个）、Q12931（porter_80）的 7c04/7c7b（24 个中 2 个）。方向=家族共享判定偏保守（漏报共享）。**特别注记**：B9W5G6 是"零同家族阳性"三之一，其零家族结论基于 14/17 PDB 有注记的集合——该结论带此注记陈述。候选 P0DP29（人工复核标记之一）反向存在 1 个新条目 3evv：09-20 家族文件知晓但 09-13 映射没有→其 n_pdb/n_studies 计数偏保守（≥44 PDB，不影响 PN-A 资格）。
3. **PubMed 缺失 102 个 PDB**（键级分布：Q9RZA4 21/60、P07900 51/450、P02829 4/49、Q12931 5/24 等）——这些条目在 pdb_pubmed（09-20）无行（无主引文链接的沉积结构）。影响：n_independent_studies 偏保守（低估），方向对 PN 门槛（n_independent≥2）安全。

**全局集偏差（J5，规模级）**：fam0920−idx0912=267、pubmed0920−idx0912=89（09-12→09-20 间新释放条目，方向正常）；entry_type=idx=259,693 条目一致。

## 3. 受影响任务清单

| 任务 | 连接 | 影响 | 处置 |
|---|---|---|---|
| L2 recall@k/enrichment | A 宇宙=fasta 2026_03 | 阳性作为查询不需在 A 内；2 TrEMBL-only 阳性的长度归一用 REST 锚 | 版本锁定例外登记 |
| L2 family-stratified | 家族层 09-20 | 阳性侧 5 PDB 无注记→分层保守 | 注记随行 |
| L3 匹配变量（长度比） | fasta(候选)/REST(阳性) | 25/25+8/10 一致；2 阳性 REST 锚 | 例外登记 |
| L3 匹配变量（家族共享） | 家族层 09-20 | 同上保守方向 | 注记随行 |
| PN 分层（n_pdb/n_studies） | 09-13 映射+09-20 pubmed | P0DP29 +3evv 漏计；102 无引文 PDB 低估 | 保守方向，登记 |
| 覆盖字段 | fasta 分母+09-13 段 | 25/25 复算一致 | 无 |

## 4. 结论（对 G1 第 12 项/建议 7 的替换文本）

跨源连接证据**足以支持有条件接受**版本差异，但旧文本"影响限于覆盖分母（cap 1.0）"不成立，替换为三项审计事实：①2/10 阳性 TrEMBL-only 无 fasta 锚（REST 锚例外）；②家族层注记缺口 5+1 例（保守方向）；③PubMed 缺失 102 例（独立研究计数保守）。接受后的约束不变（版本锁定条款），另加"例外与注记随行进入 L2/L3 报告"。

## 5. 局限
- 键级核查（35 键）；未做 A 宇宙 48 万键全量长度交叉（那将等价于重下 UniProt 2026.04 全库，超出必要）。
- 家族缺行原因（未分类 vs 快照缺口）未逐一回查 CATH/SCOP2 原库（保守方向不改变判定）。
