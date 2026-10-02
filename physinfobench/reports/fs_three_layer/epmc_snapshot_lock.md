# Europe PMC 检索快照锁定方案（P1.16 交付）

## 快照构成

1. **主快照**（P1.11，冻结）：
   - 查询式模板：`("<accession>" OR "<protein_name>") AND ("fold switching" OR "fold-switching" OR metamorphic OR "alternative fold" OR "conformational switch" OR "domain swapping")`（configs/fold_switch_putative_negative.yaml）
   - 抓取时间：2026-09-23 09:25 Asia/Shanghai（search_meta.json.query_date；主快照=09:01 首轮与 09:25 重建轮合并，56 新抓+82 复用，构成披露见 P1.11 报告）
   - 原始响应：data/raw/europepmc/2026-09-23/（216 文件：194 page1 级（138 现役候选+56 递补前候选留档）+21 分页+1 meta）
   - 校验值：data/manifests/europepmc_20260923_checksums.tsv（逐文件 SHA256）
   - 逐候选 HTTP 状态与检索日期：search_meta.json（live 写/cached 读，重放一致）
   - 复现方式：`python3 scripts/build_fs_putative_negative_candidates.py --search-mode cached ...`（三表逐字节一致，P1.11 两轮审核验证）
2. **截断回补注册表**（P1.16，2026-09-23 18:12–18:47）：
   - 对象：7 个 fetched_all_hits=false 候选（全部已在 EXITED 队列，与 PN-A 主分析分开）
   - 原始响应：data/raw/europepmc_p116_backfill/2026-09-23/（.pageN.json，N≥2；逐文件 SHA256 见下）
   - 终态（2026-09-23 20:32 终轮；此后回补冻结为只读重放，P116_BACKFILL_LIVE=1 方可推进）：5 complete（O67024=118/118、Q5JF30=118/118、P21589=52/52、P06787=797/797、Q9Y9L0=118/118）+2 partial（O16305=725/797、P60204=625/797；cursorMark 不稳定，单调推进不回退；partial 按检索覆盖=已取回页数）
   - 回补后转换类关键词复扫（最终）：P06787=43、O16305=43、P60204=41、O67024/Q5JF30/Q9Y9L0=8、P21589=1 条命中（均已在 EXITED 队列，方向一致）
3. **论文级证据存档**（P1.16）：data/raw/pna_fulltext/2026-09-23/papers/（(候选,DOI) 粒度快照 JSON）+ fullTextXML（9 个）+ MED_{pmid}.json 摘要响应 + rcsb_primary_citations.json + epmc_doi_resolution.json（快照优先语义：已解析记录与已存档全文一律复用本地，杜绝服务端抖动进入结果）。

## 锁定政策（供 G1 二选一）

- **选项 A（推荐）**：G1 确认时冻结当前快照——后续 P2.01 匹配与 L3 报告一律引用本快照；Europe PMC 索引漂移不再进入证据链。理由：快照+cached 重放已可复现；EXITED 队列方向不受 partial 影响。
- **选项 B**：P2.01 匹配前对"将进入最终匹配集的候选"重检一次并双时间点对比（敏感性），其余仍用快照。成本：一次受控重检+差异登记。

## 已知边界

- EuropePMC DOI 检索与 cursorMark 分页存在服务端不稳定（同 DOI 异答/翻页中断），本方案以"快照优先+记忆化+单调回补"消化；任何新增抓取须登记目录与校验值，不覆盖既有快照。
- "检索无命中"始终是检索结果陈述，不是可靠阴性（manual_review_status 分列）。
