#!/usr/bin/env python3
"""P5.02 步骤 0（输入构建，P5.01 lock 钉版协议）：确认集 fasta/残基域索引构建。

纪律：本脚本只做输入构建（序列+冻结存储域），不计算任何指标、不读任何模型分数；
运行时刻在 confirmation_lock 注册之后、确认集结果级 first_read 之前。
域构建依据（全部冻结文件）：
  - knots：split_manifest(task_area=knot, split=confirmation) ∩ knots_sequences.tsv（usable=已解析序列）
    ∩ knots.tsv presence_mask=1；标签 presence_target 仅写入 fasta 元数据 qc，不参与任何选择；
  - disorder：split_manifest(disorder, confirmation) 全部 432 行；分母残基域=disorder_masks
    state∈{0,1}∧mask=1（probes.yaml residue_storage_domain 冻结规则）；截断 1022 裁剪规则同
    build_p303_inputs.py（空域 EMPTY_EVALUABLE_SET 出分母并登记）；
  - FS 端点：确认 6 对 ×2 端点，序列=fold_pair_endpoint_sequence_structure_summary.tsv
    observed_sequence（与 P3.03 dev 端点同源同协议）。
输出：data/interim/p501/{knots_conf.fa, disorder_conf.fa, disorder_conf_resid_idx.tsv,
fs_conf_endpoints.fa, inputs_qc.json}；inputs_qc.json 含全部输出 sha256（集群侧校验依据）。
"""
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "data/interim/p501")
B = "/Users/yuan/Documents/Codex/2026-09-08/jie/benchmark_step1"


def die(m):
    print(f"[p501in FATAL] {m}", file=sys.stderr)
    sys.exit(1)


if not os.path.isdir(B):
    die(f"FS 端点序列源目录不存在（与 build_p303_inputs.py 同源的仓库外依赖）：{B}")


def rd(p, d="\t"):
    with open(p, newline="") as f:
        return list(csv.DictReader(f, delimiter=d))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_fa(path, items):
    with open(path, "w") as f:
        for n, s in items:
            f.write(f">{n}\n{s}\n")


os.makedirs(OUT, exist_ok=True)
man = rd(os.path.join(ROOT, "data/splits/split_manifest.tsv"))
qc = {"built_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "outputs": {}}

# ---- 1) knots 确认集（usable ∩ 有序列）----
kseq = {r["record_id"].lower(): r for r in rd(os.path.join(ROOT, "data/curated/knots_sequences.tsv"))}
kdb = json.load(open(os.path.join(ROOT, "data/raw/rcsb/2026-09-25/knot_entry_sequences.json")))
knots = {r["record_id"].lower(): r for r in rd(os.path.join(ROOT, "data/curated/knots.tsv"))}
conf_k = [r["sample_id"].split(":", 1)[1] for r in man
          if r["task_area"] == "knot" and r["split"] == "confirmation"]
items, n_pos, n_neg = [], 0, 0
missing = []
for rec in conf_k:
    krec = knots.get(rec.lower())
    if krec is None:
        die(f"确认链不在 knots.tsv: {rec}")
    if krec["presence_mask"] != "1":
        continue  # 存在性任务分母=presence_mask=1（probes.yaml 冻结口径）；type 专属行不入
    s = kseq.get(rec.lower())
    if s is None:
        missing.append(rec)
        continue
    label = krec["presence_target"]
    if label == "1":
        n_pos += 1
    elif label == "0":
        n_neg += 1
    else:
        die(f"{rec} presence_target 非法")
    e, ch = s["pdb"], s["chain"]
    if e not in kdb or ch not in kdb[e]:
        die(f"{rec} 序列库缺 {e}/{ch}")
    items.append((f"knot:{rec}", kdb[e][ch]))
if sorted(missing) != ["1giy_M", "2hfx_A"]:
    die(f"无序列确认链 {sorted(missing)} != 预期 [1giy_M, 2hfx_A]（lock 覆盖披露口径）")
if (n_pos, n_neg) != (22, 126):
    die(f"knots 确认可评价 {n_pos}/{n_neg} != lock 钉版 22/126")
qc["knots_conf"] = {"n_total_manifest": len(conf_k), "n_evaluable": len(items),
                    "n_pos": n_pos, "n_neg": n_neg,
                    "no_sequence": sorted(missing)}
write_fa(os.path.join(OUT, "knots_conf.fa"), items)

# ---- 2) disorder 确认集（432 全量；分母域+截断规则同 dev 构建器）----
dp = json.load(open(os.path.join(ROOT, "data/raw/disprot/2026-09-22/disprot_api_search.json")))
dpseq = {x["acc"]: x["sequence"] for x in dp["data"]}
id2acc = {r["disprot_id"]: r["uniprot_acc"] for r in rd(os.path.join(ROOT, "data/curated/disorder.tsv"))}
dom = defaultdict(list)
for r in rd(os.path.join(ROOT, "data/curated/disorder_masks.tsv")):
    if r["state"] in ("0", "1") and r["mask"] == "1":
        dom[r["disprot_id"]].extend(range(int(r["start"]), int(r["end"]) + 1))
conf_d = sorted(r["sample_id"].split(":", 1)[1] for r in man
                if r["task_area"] == "disorder" and r["split"] == "confirmation")
dis_items, dis_ridx = [], []
n_empty = n_beyond = n_clip = 0
for did in conf_d:
    acc = id2acc.get(did)
    seq = dpseq.get(acc)
    if not seq:
        die(f"disorder {did}/{acc} 无序列")
    idx = sorted(set(dom.get(did, [])))
    if not idx:
        n_empty += 1
        continue
    clipped = [i for i in idx if i <= 1022]
    if len(clipped) < len(idx):
        n_clip += len(idx) - len(clipped)
        idx = clipped
    if not idx:
        n_beyond += 1
        continue
    if max(idx) > len(seq):
        die(f"disorder {did}: 索引域异常")
    dis_items.append((did, seq))
    dis_ridx.append({"name": did, "resid_indices": ",".join(map(str, idx))})
if len(dis_items) + n_empty + n_beyond != 432:
    die(f"disorder 确认 432 口径破坏: {len(dis_items)}+{n_empty}+{n_beyond}")
qc["disorder_conf"] = {"n_total_manifest": 432, "n_extracted": len(dis_items),
                       "n_empty_domain": n_empty, "n_beyond_trunc": n_beyond,
                       "n_clip_residues": n_clip,
                       "n_domain_residues": sum(len(x["resid_indices"].split(",")) for x in dis_ridx)}
write_fa(os.path.join(OUT, "disorder_conf.fa"), dis_items)
with open(os.path.join(OUT, "disorder_conf_resid_idx.tsv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["name", "resid_indices"], delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(dis_ridx)

# ---- 3) FS 确认对端点（6 对 ×2；描述性 side-table 输入，非 verdict）----
summ = {(r["pdb_id"].lower(), r["requested_chain"].upper()): r["observed_sequence"]
        for r in rd(B + "/manifests/fold_pair_endpoint_sequence_structure_summary.tsv")}
fs = {r["pair_id"]: r for r in rd(os.path.join(ROOT, "data/curated/fold_switch_global.tsv"))}
fs_items = []
for r in man:
    if r["task_area"] == "fold_switch" and r["split"] == "confirmation":
        rec = fs[r["sample_id"]]
        for side, pdb, ch in (("A", rec["pdb_a"], rec["chain_a"]), ("B", rec["pdb_b"], rec["chain_b"])):
            s = summ.get((pdb.lower(), ch.upper()))
            if not s:
                die(f"FS 端点无序列 {r['sample_id']}_{side}")
            fs_items.append((f"{r['sample_id']}_{side}", s))
if len(fs_items) != 12:
    die(f"FS 确认端点 {len(fs_items)} != 12")
qc["fs_conf_endpoints"] = {"n_pairs": 6, "n_endpoints": len(fs_items)}
write_fa(os.path.join(OUT, "fs_conf_endpoints.fa"), fs_items)

# ---- 4) 输出哈希清单（集群侧校验依据）----
for fn in ["knots_conf.fa", "disorder_conf.fa", "disorder_conf_resid_idx.tsv", "fs_conf_endpoints.fa"]:
    qc["outputs"][fn] = sha256(os.path.join(OUT, fn))
qc["source_pins"] = {p: sha256(os.path.join(ROOT, p)) for p in [
    "data/splits/split_manifest.tsv", "data/curated/knots.tsv",
    "data/curated/knots_sequences.tsv", "data/curated/disorder_masks.tsv",
    "data/curated/disorder.tsv", "data/curated/fold_switch_global.tsv"]}
with open(os.path.join(OUT, "inputs_qc.json"), "w") as f:
    json.dump(qc, f, ensure_ascii=False, indent=1, sort_keys=True)
print("[p501in] OK", json.dumps({k: v for k, v in qc.items()
                                 if k in ("knots_conf", "disorder_conf", "fs_conf_endpoints")},
                                ensure_ascii=False))
