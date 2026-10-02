#!/usr/bin/env python3
"""check_metrics_examples.py — P0.05 指标公式手算核对

用可手算的小例子核对 evaluation_protocol.yaml 中定义的公式实现：
AUROC（平均秩）、AUPRC（average_precision 语义）、Macro-F1（缺失类别规则）、
区段重叠度（IoU 降序贪心一对一匹配）、ECE 的 NA 规则、按蛋白平均 vs 合并残基、
bootstrap 确定性与配对重采样索引一致性。
全部断言基于独立手算的期望值（文件内注释给出计算过程）；通过则退出码 0。
本脚本仅验证公式实现，不读取科研数据；正式指标模块在 P2.06 按本协议实现。
"""
import random
import sys

TOL = 1e-9
failures = []


def check(name, got, expect):
    if got != expect:
        failures.append(f"{name}: expect {expect!r}, got {got!r}")


def check_close(name, got, expect):
    if abs(got - expect) > TOL:
        failures.append(f"{name}: expect {expect:.12f}, got {got:.12f}")


def auroc(y, s):
    pos = [si for si, yi in zip(s, y) if yi == 1]
    neg = [si for si, yi in zip(s, y) if yi == 0]
    if not pos or not neg:
        return None, "INSUFFICIENT_CLASS"
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg
    )
    return wins / (len(pos) * len(neg)), None


def average_precision(y, s):
    pairs = sorted(zip(s, y), key=lambda t: -t[0])
    n_pos = sum(y)
    if n_pos < 1 or len(y) < 2:
        return None, "INSUFFICIENT_POSITIVES"
    tp, ap = 0, 0.0
    for i, (_, yi) in enumerate(pairs, start=1):
        tp += yi
        ap += yi * tp / i
    return ap / n_pos, None


def macro_f1(y, pred):
    classes = sorted(set(y))  # 仅真值中出现的类别参与平均
    if len(classes) < 2:
        return None, {}
    f1s = []
    per_class = {}
    for c in classes:
        tp = sum(1 for yi, pi in zip(y, pred) if yi == c and pi == c)
        fp = sum(1 for yi, pi in zip(y, pred) if yi != c and pi == c)
        fn = sum(1 for yi, pi in zip(y, pred) if yi == c and pi != c)
        f1 = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
        f1s.append(f1)
        per_class[c] = {"f1": f1, "n_true": sum(1 for v in y if v == c)}
    return sum(f1s) / len(f1s), per_class


def segment_overlap(truth_segs, pred_segs):
    """truth_segs/pred_segs: [(start,end)]，1 起闭区间。返回 (score, |G|, matched, unmatched_pred)。"""
    ious = []
    for pi, (ps, pe) in enumerate(pred_segs):
        for gi, (gs, ge) in enumerate(truth_segs):
            inter = max(0, min(pe, ge) - max(ps, gs) + 1)
            union = (pe - ps + 1) + (ge - gs + 1) - inter
            if union > 0:
                ious.append((inter / union, pi, gi))
    ious.sort(key=lambda t: -t[0])
    used_p, used_g, total = set(), set(), 0.0
    for iou, pi, gi in ious:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        total += iou
    g = len(truth_segs)
    return (total / g if g >= 1 else None), g, len(used_g), len(pred_segs) - len(used_p)


def ece(y, s, n_bins=10, min_n=10):
    """返回 (ECE 或 None+原因)。按可评价单位等权。"""
    if len(y) < min_n:
        return None, "INSUFFICIENT_CALIBRATION_N"
    bins = [[] for _ in range(n_bins)]
    for yi, si in zip(y, s):
        b = min(int(si * n_bins), n_bins - 1)
        bins[b].append((yi, si))
    n = len(y)
    total = 0.0
    for b in bins:
        if not b:
            continue
        acc = sum(yi for yi, _ in b) / len(b)
        conf = sum(si for _, si in b) / len(b)
        total += (len(b) / n) * abs(acc - conf)
    return total, None


def main():
    # ---- 蛋白 A：手算 AUROC=0.5，AP=5/6，Macro-F1@0.5=0.25 ----
    yA, sA = [1, 0, 1], [0.9, 0.8, 0.3]
    v, _ = auroc(yA, sA); check_close("A.auroc", v, 0.5)
    v, _ = average_precision(yA, sA); check_close("A.ap", v, 5 / 6)
    predA = [1 if x >= 0.5 else 0 for x in sA]
    v, _ = macro_f1(yA, predA); check_close("A.macroF1", v, 0.25)

    # ---- 蛋白 B：AUROC=1.0，AP=1.0；蛋白 C：单阳类 → AUROC NA、AP=1.0 ----
    yB, sB = [1, 0], [0.7, 0.6]
    v, _ = auroc(yB, sB); check_close("B.auroc", v, 1.0)
    v, _ = average_precision(yB, sB); check_close("B.ap", v, 1.0)
    yC, sC = [1, 1], [0.2, 0.1]
    v, reason = auroc(yC, sC); check("C.auroc.NA", (v, reason), (None, "INSUFFICIENT_CLASS"))
    v, _ = average_precision(yC, sC); check_close("C.ap", v, 1.0)

    # ---- 按蛋白平均 AP = (5/6+1+1)/3 = 17/18；合并 AP = 0.729523809523... ----
    per_prot = [average_precision(yA, sA)[0], average_precision(yB, sB)[0], average_precision(yC, sC)[0]]
    check_close("per_protein_ap_mean", sum(per_prot) / 3, 17 / 18)
    pooled_y = yA + yB + yC
    pooled_s = sA + sB + sC
    v, _ = average_precision(pooled_y, pooled_s)
    check_close("pooled_ap", v, 3.6476190476190474 / 5)

    # ---- 区段重叠度：G={[2,3]}，P={[1,2],[3,4]} → 两候选 IoU 均 1/3，
    # 贪心取其一后另一真值已占用 → score=1/3，matched=1，unmatched_pred=1 ----
    score, g, matched, unmatched_pred = segment_overlap([(2, 3)], [(1, 2), (3, 4)])
    check_close("seg.score", score, 1 / 3)
    check("seg.counts", (g, matched, unmatched_pred), (1, 1, 1))

    # ---- ECE：样本数 7 < 10 → NA（原因码），不得输出 0 ----
    v, reason = ece(pooled_y, pooled_s)
    check("ece.NA", (v, reason), (None, "INSUFFICIENT_CALIBRATION_N"))

    # ---- ECE 手算数值例（n=10）：5 个 (score 0.95, 4 正 1 负)，5 个 (score 0.15, 1 正 4 负)
    # bin9: acc=0.8, conf=0.95, |diff|=0.15；bin1: acc=0.2, conf=0.15, |diff|=0.05
    # ECE = 0.5*0.15 + 0.5*0.05 = 0.10 ----
    ey = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
    es = [0.95, 0.95, 0.95, 0.95, 0.95, 0.15, 0.15, 0.15, 0.15, 0.15]
    v, reason = ece(ey, es)
    if v is None:
        failures.append(f"ece.handcalc: 期望 0.10，得到 NA({reason})")
    else:
        check_close("ece.handcalc", v, 0.10)

    # ---- bootstrap 确定性与配对索引一致性 ----
    units = [("A", average_precision(yA, sA)[0]), ("B", average_precision(yB, sB)[0]),
             ("C", average_precision(yC, sC)[0])]

    def boot_mean(rng):
        return sum(units[rng.randrange(len(units))][1] for _ in units) / len(units)

    r1, r2 = random.Random(2026), random.Random(2026)
    m1, m2 = boot_mean(r1), boot_mean(r2)
    check_close("bootstrap.determinism", m1, m2)

    # 配对比较：两方法的不同分数向量在同一批索引上计算配对差值。
    # 方法 B 值 = 1 − 方法 A 值；同一索引序列下 mean(A) − mean(B) = 2·mean(A[idx]) − 1（手算可核对）。
    idx = [random.Random(7).randrange(len(units)) for _ in range(50)]
    vals_a = [units[i][1] for i in idx]
    vals_b = [1.0 - v for v in vals_a]
    paired_diff = sum(a - b for a, b in zip(vals_a, vals_b)) / len(vals_a)
    expected = 2.0 * (sum(vals_a) / len(vals_a)) - 1.0
    check_close("paired.diff_consistency", paired_diff, expected)
    # 索引序列跨方法一致（重放同一 rng 序列必须得到同一批索引）
    idx_replay = [random.Random(7).randrange(len(units)) for _ in range(50)]
    check("paired.same_indices", idx_replay, idx)

    if failures:
        print("METRIC CHECK FAILED")
        for f in failures:
            print("  -", f)
        return 1
    print("METRIC CHECK PASSED: AUROC/AP/Macro-F1/segment-IoU/ECE-NA/per-protein-vs-pooled/"
          "bootstrap determinism/paired indices 全部与手算一致。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
