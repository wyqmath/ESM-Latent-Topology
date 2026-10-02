#!/usr/bin/env python3
"""P5.07 修复：终版宇宙构建器（本地，消费全部边源后一次性产出）。

输入：
  - p507_type_frozen_table.tsv（冻结类型表 + 状态列）
  - p507_isolate/seq_clusters_rep.tsv（mmcss 簇表，逐行 rep<TAB>member）
  - p507_isolate/usalign_edges_v2.tsv（双向确认的新-旧结构边）
  - p507_isolate/newnew_edges_v2.tsv（双向确认的新-新结构边）
  - split_manifest.tsv（旧链 dev_fold + 冲突检测）
输出：data/interim/p507_final_universe_20261002.json（本次双向确认版；旧宇宙保留）
规则（预注册 p507_type_probe_design.yaml）：
  - 分量=相同簇/结构边之并；分量原子进出 LCO；
  - 分量触及 confirmation/final_holdout（序列簇共属或结构边）→ 整分量剔除；
  - 节点 store：新=p507_emb 的 `p507:CHAIN`，旧=knots_resid 的 `knot:CHAIN`。
"""
import csv
import json
import os
import math
from collections import defaultdict, Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
I = lambda *p: os.path.join(ROOT, "data", "interim", *p)


def expected_pairs(path, *, newnew):
    pairs = set()
    with open(path) as source:
        for line in source:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            q = p[0].rsplit(".pdb", 1)[0]
            t = p[1].rsplit(".pdb", 1)[0]
            if newnew and q >= t:
                continue
            if max(float(p[2]), float(p[3])) >= 0.45:
                pairs.add((q, t))
    return pairs


def read_tsv(path):
    with open(path) as source:
        return list(csv.DictReader(source, delimiter="\t"))


def main():
    frozen = [t for t in read_tsv(I("p507_type_frozen_table.tsv")) if t["status"] == "frozen"]
    nodes = {}
    for t in frozen:
        nodes[t["chain"]] = {"type": t["frozen_type"], "source": "new", "store": "new"}

    knots = {r["record_id"].upper(): r for r in read_tsv(os.path.join(ROOT, "data/curated/knots.tsv"))}
    man = read_tsv(os.path.join(ROOT, "data/splits/split_manifest.tsv"))
    man_split, dev_fold = {}, {}
    for r in man:
        if r["task_area"] == "knot":
            ch = r["sample_id"].split(":", 1)[1].upper()
            man_split[ch] = r["split"]
            if r["split"] == "development":
                dev_fold[ch] = r["dev_fold"]
    kseq = {r["record_id"].upper()
            for r in read_tsv(os.path.join(ROOT, "data/curated/knots_sequences.tsv"))}
    for r in man:
        if r["task_area"] == "knot" and r["split"] == "development":
            orig = r["sample_id"].split(":", 1)[1]
            rec = knots.get(orig.upper())
            if rec and rec["type_task_tier"] == "eligible" and rec["record_id"].upper() in kseq:
                ch = orig.upper()
                nodes[ch] = {"type": rec["c2_primary"], "source": "old_dev", "store": "old",
                             "sid_manifest": orig, "dev_fold": dev_fold.get(orig)}

    parent = {c: c for c in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 边 1：mmseqs 簇共属（逐行 rep<TAB>member；簇内成员两两连通经 rep 中转）
    members_of = defaultdict(list)
    with open(I("p507_isolate", "seq_clusters_rep.tsv")) as source:
        for line in source:
            p = line.rstrip("\n").split("\t")
            if len(p) == 2:
                members_of[p[0]].append(p[1])
    for rep, ms in members_of.items():
        in_nodes = [m for m in ms if m in nodes]
        if rep in nodes:
            in_nodes.append(rep)
        for m in in_nodes[1:]:
            union(in_nodes[0], m)

    # 边 2/3：结构边（新-旧 / 新-新）
    for fn in ("usalign_edges_v2.tsv", "newnew_edges_v2.tsv"):
        fp = I("p507_isolate", fn)
        if not os.path.exists(fp):
            raise FileNotFoundError(f"缺少双向结构确认表 {fn}；拒绝生成最终宇宙")
        seen = set()
        for row in read_tsv(fp):
            q, t = row["query"], row["target"]
            if row["returncode"] != "0":
                raise ValueError(f"US-align 失败或分数不完整 {fn}: {q},{t}")
            s1, s2, tm = float(row["tm_norm_input1"]), float(row["tm_norm_input2"]), float(row["tm_min"])
            if not all(math.isfinite(x) and 0 <= x <= 1 for x in (s1, s2, tm)):
                raise ValueError(f"US-align 分数越界 {fn}: {q},{t}")
            if not math.isclose(tm, min(s1, s2), abs_tol=0.00011):
                raise ValueError(f"tm_min 与两个方向不一致 {fn}: {q},{t}")
            if (q, t) in seen:
                raise ValueError(f"重复结构配对 {fn}: {q},{t}")
            seen.add((q, t))
            if tm >= 0.6 and q in nodes and t in nodes:
                union(q, t)
        hit_name = "nn_hits.tsv" if fn.startswith("newnew_") else "foldseek_hits.tsv"
        expected = expected_pairs(I("p507_isolate", hit_name), newnew=fn.startswith("newnew_"))
        if seen != expected:
            raise ValueError(f"{fn} 的确认配对覆盖不完整：缺 {len(expected-seen)}，多 {len(seen-expected)}")

    # 冲突分量：触及 conf/holdout（序列簇共属 或 新-旧结构边）
    touch_conf = set()
    for rep, ms in members_of.items():
        if not any(m in nodes for m in ms) and rep not in nodes:
            continue
        if any(man_split.get(m) in ("confirmation", "final_holdout") for m in ms):
            for m in ms:
                if m in nodes:
                    touch_conf.add(find(m))
            if rep in nodes:
                touch_conf.add(find(rep))
    for row in read_tsv(I("p507_isolate", "usalign_edges_v2.tsv")):
        if float(row["tm_min"]) >= 0.6 and row["query"] in nodes \
                and man_split.get(row["target"]) in ("confirmation", "final_holdout"):
            touch_conf.add(find(row["query"]))

    keep = {c for c in nodes if find(c) not in touch_conf}
    out_nodes = {}
    for c in sorted(keep):
        if nodes[c]["store"] == "new":
            sid = f"p507:{c}"
        else:
            sid = f"knot:{nodes[c]['sid_manifest']}"
        out_nodes[sid] = {"type": nodes[c]["type"], "source": nodes[c]["source"],
                          "store": nodes[c]["store"], "component": find(c),
                          "dev_fold": nodes[c].get("dev_fold")}
    dropped = sorted(set(nodes) - keep)
    print(f"节点 {len(nodes)} → 冲突分量剔除后 {len(out_nodes)}（丢弃 {len(dropped)}）")
    print("类型分布:", dict(Counter(v["type"] for v in out_nodes.values())))
    print("独立分量:", len({v["component"] for v in out_nodes.values()}))
    for t in ("3_1", "4_1", "5_2"):
        tc = {v["component"] for v in out_nodes.values() if v["type"] == t}
        n_chains = sum(1 for v in out_nodes.values() if v["type"] == t)
        print(f"  {t}: 链 {n_chains} | 分量 {len(tc)}")
    output_path = I("p507_final_universe_20261002.json")
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w") as output:
        json.dump({"nodes": out_nodes, "dropped_conflict": dropped,
                   "n_components": len({v["component"] for v in out_nodes.values()})},
                  output, indent=1, sort_keys=True)
        output.write("\n")
    os.replace(tmp_path, output_path)
    print("saved p507_final_universe_20261002.json")


if __name__ == "__main__":
    main()
