#!/usr/bin/env python3
"""P2.06：统一指标计算器（正式模块；公式对齐 evaluation_protocol，P0.05 手算例为回归基准）。

实现并自检：AUROC、AUPRC（average_precision 语义）、ECE（等宽 10 分箱、按蛋白等权）、
Macro-F1（缺失类别按协议规则）、残基 AUPRC（per-protein 主口径）、区段 IoU（降序贪心一对一）、
recall_at_k / enrichment_at_k / positive_rank_percentile（L2 排序指标）。
"""
import numpy as np


def auroc(scores, labels):
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # 处理并列：平均秩
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def average_precision(scores, labels):
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    if s.ndim != 1 or y.ndim != 1 or len(s) != len(y):
        raise ValueError("scores and labels must be equal-length 1D arrays")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must contain only 0 and 1")
    n_pos = int(y.sum())
    if not n_pos:
        return float("nan")
    # sklearn average_precision_score semantics: score ties share one precision.
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    tp = 0
    total = 0.0
    end = 0
    for score in np.unique(s_sorted)[::-1]:
        start = end
        while end < len(s_sorted) and s_sorted[end] == score:
            end += 1
        group_pos = int(y_sorted[start:end].sum())
        tp += group_pos
        total += (tp / end) * group_pos
    return float(total / n_pos)


def ece_protein_equal_weight(probs, labels, protein_ids, bins=10):
    df = list(zip(np.asarray(probs, float), np.asarray(labels, int), protein_ids))
    by_p = {}
    for p, l, pid in df:
        by_p.setdefault(pid, []).append((p, l))
    ces = []
    for pid, items in by_p.items():
        items.sort(key=lambda x: x[0])
        n = len(items)
        ce = 0.0
        for b in range(bins):
            lo, hi = b / bins, (b + 1) / bins
            grp = [x for x in items if (lo <= x[0] < hi) or (b == bins - 1 and lo <= x[0] <= hi)]
            if not grp:
                continue
            conf = sum(x[0] for x in grp) / len(grp)
            acc = sum(x[1] for x in grp) / len(grp)
            ce += len(grp) / n * abs(conf - acc)
        ces.append(ce)
    return float(np.mean(ces))


def macro_f1(preds, labels, classes=None):
    """协议 missing_class 规则：仅在真值中出现的类别参与平均；真值存在但预测恒缺时 F1=0 计入。"""
    classes = sorted(set(np.asarray(labels).tolist())) if classes is None else classes
    f1s = []
    for c in classes:
        tp = sum(1 for p, l in zip(preds, labels) if p == c and l == c)
        fp = sum(1 for p, l in zip(preds, labels) if p == c and l != c)
        fn = sum(1 for p, l in zip(preds, labels) if p != c and l == c)
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f1s))


def residue_auprc_per_protein(scores_by_protein, labels_by_protein):
    aps = []
    for pid in scores_by_protein:
        aps.append(average_precision(scores_by_protein[pid], labels_by_protein[pid]))
    return float(np.mean(aps))


def segment_iou(pred_segments, gold_segments):
    """降序贪心一对一匹配；返回平均 IoU（协议口径）。"""
    cands = []
    for p in pred_segments:
        for g in gold_segments:
            inter = max(0, min(p[1], g[1]) - max(p[0], g[0]) + 1)
            union = (p[1] - p[0] + 1) + (g[1] - g[0] + 1) - inter
            if inter > 0:
                cands.append((inter / union, p, g))
    cands.sort(key=lambda x: -x[0])
    used_p, used_g, ious = set(), set(), []
    for iou, p, g in cands:
        if p in used_p or g in used_g:
            continue
        used_p.add(p)
        used_g.add(g)
        ious.append(iou)
    return float(np.mean(ious)) if ious else 0.0


def recall_at_k(ranked_ids, positive_ids, k):
    top = ranked_ids[:k]
    return len(set(top) & set(positive_ids)) / len(positive_ids) if positive_ids else float("nan")


def enrichment_at_k(ranked_ids, positive_ids, k, universe_size):
    hit = len(set(ranked_ids[:k]) & set(positive_ids))
    base = len(positive_ids) / universe_size
    return (hit / k) / base if base > 0 else float("nan")


def positive_rank_percentile(ranked_ids, positive_ids):
    """正例平均排名除以总数（越小越靠前，与 PLM 直推结果方向一致）。"""
    pos_rank = [i + 1 for i, x in enumerate(ranked_ids) if x in set(positive_ids)]
    return float(np.mean(pos_rank) / len(ranked_ids)) if pos_rank else float("nan")


if __name__ == "__main__":
    # 手算小例（对齐 P0.05 校验脚本的教学例）
    assert abs(auroc([0.9, 0.8, 0.7, 0.1], [1, 0, 1, 0]) - 0.75) < 1e-9
    assert abs(average_precision([0.9, 0.8, 0.7, 0.1], [1, 0, 1, 0]) - 5 / 6) < 1e-9
    assert average_precision([0.5, 0.5], [1, 0]) == 0.5
    assert average_precision([0.5, 0.5], [0, 1]) == 0.5
    assert macro_f1([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert abs(macro_f1([1, 1, 1, 1], [1, 0, 0, 0]) - 0.2) < 1e-9  # F1_1=0.4, F1_0=0（真值存在预测恒缺）→ 0.2
    assert macro_f1([1, 1, 1], [1, 1, 1]) == 1.0  # 类 0 真值缺席 → 不参与
    assert abs(segment_iou([(1, 10)], [(1, 10)]) - 1.0) < 1e-9
    assert abs(segment_iou([(1, 10)], [(6, 15)]) - 1 / 3) < 1e-9  # inter=5, union=15
    assert recall_at_k(list("abcd"), ["a"], 2) == 1.0  # top2 含唯一正例
    assert abs(enrichment_at_k(list("abc"), ["a"], 1, 3) - 3.0) < 1e-9  # precision@1(=1)/base(=1/3)=3
    ranked = ["a", "b", "c", "d"]
    assert abs(positive_rank_percentile(ranked, ["a"]) - 0.25) < 1e-9
    assert abs(positive_rank_percentile(ranked, ["d"]) - 1.0) < 1e-9
    e = ece_protein_equal_weight([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0], ["p", "p", "q", "q"])
    assert abs(e - 0.0) < 1e-9  # 完美校准
    e2 = ece_protein_equal_weight([0.9, 0.9, 0.1, 0.1], [1, 1, 0, 0], ["p", "p", "q", "q"])
    assert abs(e2 - 0.1) < 1e-9  # bin [0.9,1) 内 conf=0.9/acc=1.0
    print("metrics module: all hand-check assertions passed")
