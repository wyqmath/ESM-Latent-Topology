#!/usr/bin/env python3
"""Confirm Foldseek new-chain pairs with both US-align TM-score normalizations."""
import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from usalign_parser import parse_usalign_scores


FIELDS = ["query", "target", "tm_min", "tm_norm_input1", "tm_norm_input2", "returncode"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--usalign", required=True)
    args = ap.parse_args()
    hit_path = os.path.join(args.out, "nn_hits.tsv")
    out_path = os.path.join(args.out, "newnew_edges_v2.tsv")

    candidates = set()
    with open(hit_path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            q = p[0].rsplit(".pdb", 1)[0]
            t = p[1].rsplit(".pdb", 1)[0]
            if q >= t or max(float(p[2]), float(p[3])) < 0.45:
                continue
            candidates.add((q, t))

    completed = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                try:
                    if row["returncode"] != "0":
                        continue
                    s1, s2, tm = (float(row[k]) for k in
                                  ("tm_norm_input1", "tm_norm_input2", "tm_min"))
                    if abs(tm - min(s1, s2)) <= 0.00011:
                        completed[(row["query"], row["target"])] = row
                except (ValueError, KeyError, TypeError):
                    continue
    pending = sorted(candidates - completed.keys())
    print(f"US-align 待确认 {len(pending)} / 候选 {len(candidates)}", flush=True)

    def work(pair):
        q, t = pair
        try:
            result = subprocess.run(
                [args.usalign, f"{args.out}/pdb_new/{q}.pdb", f"{args.out}/pdb_new/{t}.pdb", "-m", "true"],
                capture_output=True, text=True, timeout=120)
            if result.returncode:
                return dict(query=q, target=t, tm_min="", tm_norm_input1="",
                            tm_norm_input2="", returncode=result.returncode)
            s1, s2, tm = parse_usalign_scores(result.stdout)
            return dict(query=q, target=t, tm_min=f"{tm:.6f}", tm_norm_input1=f"{s1:.6f}",
                        tm_norm_input2=f"{s2:.6f}", returncode=0)
        except Exception as exc:
            print(f"[US-align failed] {q} {t}: {exc}", file=sys.stderr)
            return dict(query=q, target=t, tm_min="", tm_norm_input1="",
                        tm_norm_input2="", returncode=1)

    with ThreadPoolExecutor(16) as pool:
        for row in pool.map(work, pending):
            completed[(row["query"], row["target"])] = row
    tmp = out_path + ".tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(completed[pair] for pair in sorted(completed))
    os.replace(tmp, out_path)
    bad = [pair for pair in sorted(candidates) if pair not in completed
           or completed[pair]["returncode"] != 0]
    print(f"双向确认成功 {len(candidates)-len(bad)} / {len(candidates)}；最小 TM≥0.6 "
          f"{sum(float(completed[p]['tm_min']) >= .6 for p in candidates if p not in bad)}")
    if bad:
        raise SystemExit(f"仍有 {len(bad)} 对未确认；已保留成功结果，可重投后续跑")


if __name__ == "__main__":
    main()
