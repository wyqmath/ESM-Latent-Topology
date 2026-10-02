#!/usr/bin/env python3
"""P1.20 strict 证据强度分列（G1 材料修订第 2 项）。

把两个维度分列：
  维度一（序列身份，P1.13）：sequence_identity_review.tsv——同蛋白判定、两端观测序列
    是否完全相同（派生规则：两端 identity=1.0 且无内部缺段/错配且 obs_len 与 canon_flank
    完全一致）、构建体差异明细、isoform 竞争状态。
  维度二（双态原文证据，P1.14/round2）：dual_state_evidence_review.tsv 端点级 evidence_level；
    不在表内的 strict 对走 P1.02 导入的 round2 通道（旧项目历史审计，本项目未自核）。

10 对统一为 evidence_tiered_candidate；不产生"10/10 双态原文均已独立确认"类合并表述。
"""
import csv
import datetime
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEQ = os.path.join(ROOT, "reports/fs_three_layer/sequence_identity_review.tsv")
DUAL = os.path.join(ROOT, "reports/fs_three_layer/dual_state_evidence_review.tsv")
GLOB = os.path.join(ROOT, "data/curated/fold_switch_global.tsv")
OUT = os.path.join(ROOT, "reports/fs_three_layer/strict_evidence_tiers.tsv")
QC = os.path.join(ROOT, "reports/fs_three_layer/strict_evidence_tiers_qc.json")
RUN_TS = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def die(m):
    print(f"[P1.20 FATAL] {m}", file=sys.stderr)
    sys.exit(1)


def rd(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


strict = {r["pair_id"]: r for r in rd(GLOB) if r["tier"] == "strict_state_candidate"}
if len(strict) != 10:
    die(f"strict 行数={len(strict)} != 10")

seq_rows = rd(SEQ)
dual_rows = rd(DUAL)
if len(seq_rows) != 20:
    die(f"sequence_identity_review 行数={len(seq_rows)} != 20")

by_pair_seq = {p: [] for p in strict}
for r in seq_rows:
    if r["pair_id"] in by_pair_seq:
        by_pair_seq[r["pair_id"]].append(r)
dual_by_pair = {p: {} for p in strict}
for r in dual_rows:
    if r["pair_id"] in dual_by_pair:
        dual_by_pair[r["pair_id"]][r["endpoint"]] = r


def kw_present(notes):
    if not notes:
        return "kw_absent"
    for k in ("abstract_kw=", "kw="):
        if k in notes:
            return "kw_present"
    return "kw_absent"


def iso_status(rows):
    notes = " ".join(r.get("isoform_competition_note") or "" for r in rows).strip()
    if not notes:
        return "no_isoform_note"
    if "并列" in notes or "tie" in notes.lower():
        return "isoform_tie_sifts_attribution"
    if "canonical" in notes and ("更优" in notes or "严格" in notes):
        return "canonical_strictly_better"
    return "isoform_noted"


def pair_sides(pid):
    """pair_id -> (endpoint_key_A, endpoint_key_B)，与 fold_switch_global 的 pdb_a/chain_a、pdb_b/chain_b 对应。"""
    left, right = pid.split("__")
    a_ep, b_ep = left.rsplit("_", 1)[1], right
    return (f"{a_ep[:4]}_{a_ep[4:]}", f"{b_ep[:4]}_{b_ep[4:]}")


out_rows = []
tier_counts = {}
for pid in sorted(strict, key=lambda x: int(x.split("_")[1])):
    s = sorted(by_pair_seq[pid], key=lambda r: r["endpoint"])
    if len(s) != 2:
        die(f"{pid} 端点数={len(s)} != 2")
    a, b = s
    ep_a_key, ep_b_key = pair_sides(pid)
    side_of = {r["endpoint"]: ("A" if r["endpoint"] == ep_a_key else "B") for r in (a, b)}
    if set(side_of.values()) != {"A", "B"}:
        die(f"{pid} 端点键无法映射到 pair 侧: {ep_a_key}/{ep_b_key} vs {[r['endpoint'] for r in s]}")
    # 维度一派生
    ident_ok = all(r["interpretation"].startswith("same_protein") for r in (a, b))
    clean = all(float(r["identity_in_aligned"]) == 1.0 and int(r["internal_gap_blocks"]) == 0
                and int(r["mismatch_cols"]) == 0 for r in (a, b))
    same_obs = (clean and a["obs_len"] == b["obs_len"]
                and (a["canon_flank_n"], a["canon_flank_c"]) == (b["canon_flank_n"], b["canon_flank_c"]))
    diff_bits = []
    for r in s:
        side = side_of[r["endpoint"]]
        seg = [f"{side}:obs{r['obs_len']}aa/canon{r['canonical_len']}aa"]
        if int(r["canon_flank_n"]) or int(r["canon_flank_c"]):
            seg.append(f"canon侧翼N{r['canon_flank_n']}/C{r['canon_flank_c']}")
        if int(r["internal_gap_blocks"]):
            seg.append(f"内部缺段{r['internal_gap_blocks']}块/{r['internal_gap_obs_res']}aa")
        if int(r["mismatch_cols"]):
            seg.append(f"错配{r['mismatch_cols']}列")
        if float(r["cov_obs"]) < 1.0:
            seg.append(f"cov{r['cov_obs']}")
        diff_bits.append(",".join(seg))
    # 维度二
    eps = dual_by_pair[pid]
    if eps:
        levels = {e: r["evidence_level"] for e, r in eps.items()}
        kws = {e: kw_present(r["notes"]) for e, r in eps.items()}
        source = "P1.14_self_review"
        n_full = sum(1 for v in levels.values() if v.startswith("fulltext_self_verified"))
        n_abs = sum(1 for v in levels.values() if v.startswith("abstract_self_verified"))
        if n_full == 2:
            tier = "fulltext_both"
            dep = "none_self_verified"
        elif n_full == 1 and n_abs == 1:
            tier = "fulltext_one_abstract_one"
            dep = "partial_one_endpoint_abstract"
        elif n_abs == 2:
            tier = "abstract_both"
            dep = "partial_state_semantics_on_old_pointers"
        else:
            die(f"{pid} 端点证据级无法归类: {levels}")
    else:
        g = strict[pid]
        if g["evidence_endpoints_covered"] != "A+B":
            die(f"{pid} 无 P1.14 行但 round2 覆盖={g['evidence_endpoints_covered']}")
        tier = "historical_fulltext_old_audit"
        dep = "full_round2_import_not_reverified"
        levels = {"(round2)": g["evidence_level"]}
        kws = {}
        source = "P1.02_round2_import"
    tier_counts[tier] = tier_counts.get(tier, 0) + 1
    l1 = []
    if same_obs:
        l1.append("同序列双态：状态差异不可归因序列差异（全 10 对中唯一）")
    else:
        l1.append("观测序列/覆盖不同：构建体差异须作 L1 混杂协变量；状态间差异解释受限")
    if iso_status((a, b)) != "no_isoform_note":
        l1.append("isoform 竞争登记（归属沿用 SIFTS）")
    limit = ("无序列混杂（唯一同序列双态对）" if same_obs
             else "结论须限定于'同蛋白双态、构建体差异已解释但未被实验平衡'")
    out_rows.append({
        "pair_id": pid,
        "seq_identity_verdict": "same_protein_confirmed" if ident_ok else "FAIL",
        "same_observed_sequence": "yes" if same_obs else "no",
        "construct_difference_detail": " | ".join(diff_bits),
        "isoform_status": iso_status((a, b)),
        "dual_state_tier": tier,
        "endpoint_levels": ";".join(f"{e}:{v}" for e, v in sorted(levels.items())),
        "endpoint_kw": ";".join(f"{e}:{v}" for e, v in sorted(kws.items())) or "-",
        "old_audit_dependency": dep,
        "evidence_source": source,
        "l1_confound_items": "; ".join(l1),
        "conclusion_limit": limit,
        "candidate_status": "evidence_tiered_candidate",
    })

# 断言
same = [r["pair_id"] for r in out_rows if r["same_observed_sequence"] == "yes"]
if same != ["porter_87_2k0qA__2lelA"]:
    die(f"同序列双态 != 仅 porter_87: {same}")
if any(r["seq_identity_verdict"] != "same_protein_confirmed" for r in out_rows):
    die("存在序列身份未确认的 strict 对")
iso_dist = {v: sum(1 for r in out_rows if r["isoform_status"] == v)
            for v in {r["isoform_status"] for r in out_rows}}
if iso_dist != {"no_isoform_note": 7, "canonical_strictly_better": 1,
                "isoform_tie_sifts_attribution": 2}:
    die(f"isoform 三态分布异常（预期 7/1/2）: {iso_dist}")
kw0 = sum(1 for r in out_rows for part in r["endpoint_kw"].split(";")
          if part.endswith("kw_absent"))
qc = {"run_ts": RUN_TS, "pairs": 10, "tier_counts": tier_counts,
      "same_observed_only": same, "kw_absent_abstract_endpoints": kw0,
      "isoform_status_dist": {v: sum(1 for r in out_rows if r["isoform_status"] == v)
                              for v in {r["isoform_status"] for r in out_rows}}}
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0]), delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(out_rows)
with open(QC, "w") as f:
    json.dump(qc, f, ensure_ascii=False, indent=1, sort_keys=True)
print(f"[P1.20] 10 对分列完成 tier={tier_counts} same_obs={same} kw0_endpoints={kw0}")
