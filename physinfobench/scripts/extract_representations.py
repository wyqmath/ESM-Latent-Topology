#!/usr/bin/env python3
"""P2.05/P3.03：多层表示提取（ESM-2 650M，P2.04 已验证模型）v2。

v2 变更（P3.03，冻结于 configs/probes.yaml）：
  - --mean-layers：仅存指定隐藏层的池化均值（1-based；缺省=全部 34 层，兼容 P2.05 语义）；
  - --resid-layers：指定层的逐残基矩阵 fp16（缺省不存）；
  - --resid-index-file：按样本限制残基存储域（任务分母残基域，name→逗号分隔 1-based 索引）；
  - 长度排序批量前向（--batch-max-tokens token 预算）+ padding 掩码均值；
  - 存储 fp16（PREP.storage 入 prep_hash，旧缓存天然失配→重提取，不静默复用）。
缓存键 = revision + sequence_sha256 + layer_set + prep_hash；命中前核验 meta，失配=REJECT。
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import yaml
import tempfile
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "facebook/esm2_t33_650M_UR50D"
MAXLEN = 1022
PREP = {"prep": "trunc1022", "truncation": "head", "bos_eos": "strip",
        "forward_precision": "fp32", "storage": "fp16"}


def prep_hash(prep=PREP):
    return hashlib.sha256(json.dumps(prep, sort_keys=True).encode()).hexdigest()[:12]


def die(message):
    print(f"[extract FATAL] {message}", file=sys.stderr)
    raise SystemExit(1)


def read_fasta(p):
    names, seqs, name = [], {}, None
    with open(p) as source:
        for line in source:
            if line.startswith(">"):
                name = line[1:].strip()
                if not name or name in seqs:
                    raise ValueError(f"FASTA header missing or duplicated: {name!r}")
                names.append(name)
                seqs[name] = ""
            else:
                if name is None:
                    raise ValueError("FASTA sequence appears before its first header")
                seqs[name] += line.strip()
    return names, seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mean-layers", default="", help="逗号分隔 1-based 隐藏层；空=全部 34 层")
    ap.add_argument("--resid-layers", default="", help="逗号分隔 1-based 隐藏层；空=不存逐残基")
    ap.add_argument("--store-last-resid", action="store_true", help="存末层逐残基（P2.05 语义）")
    ap.add_argument("--resid-index-file", default="",
                    help="TSV: name<TAB>逗号分隔 1-based 残基索引（限制残基存储域；缺省=全长）")
    ap.add_argument("--batch-max-tokens", type=int, default=8192)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--pack-matrix", default="",
                    help="额外输出单个 npz：X fp16 (n,K,D) + names json（与逐样本 npz 并行产出）")
    ap.add_argument("--append-manifest", action="store_true",
                    help="合并同配置的分批提取清单；任何模型/层/精度/设备差异均报错")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 8)

    default_mean_layers = not args.mean_layers
    mean_layers = sorted(int(x) for x in args.mean_layers.split(",") if x.strip())
    if mean_layers and (min(mean_layers) < 1 or max(mean_layers) > 33):
        ap.error("--mean-layers 接受 hidden layer 1..33；省略时提取 hidden_states 0..33 全部 34 层")
    if not mean_layers:
        mean_layers = list(range(34))
    resid_layers = sorted(int(x) for x in args.resid_layers.split(",") if x.strip())
    resid_idx = {}
    if args.resid_index_file:
        for row in csv.DictReader(open(args.resid_index_file), delimiter="\t"):
            resid_idx[row["name"]] = [int(x) for x in row["resid_indices"].split(",") if x.strip()]

    layer_set = "mean_" + ("all34" if default_mean_layers else "_".join(map(str, mean_layers)))
    if args.store_last_resid:
        layer_set += "+last_resid"
    if resid_layers:
        layer_set += "+resid_" + "_".join(map(str, resid_layers))
    layer_set += "|dom" + ("idx" if resid_idx else "full")

    with open(os.path.join(os.path.dirname(__file__), "..", "configs", "probes.yaml")) as cfg:
        encoder = yaml.safe_load(cfg)["encoder"]
    revision = encoder["revision"]
    requested_revision = os.environ.get("ESM_REVISION_SNAPSHOT")
    if requested_revision and requested_revision != revision:
        die(f"ESM_REVISION_SNAPSHOT={requested_revision} 与冻结配置 revision={revision} 不同")
    if encoder.get("forward_precision") != PREP["forward_precision"]:
        die("PREP 前向精度与 configs/probes.yaml 不一致")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=revision)
    model = AutoModel.from_pretrained(MODEL, revision=revision,
                                      output_hidden_states=True, add_pooling_layer=False)
    loaded_revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    if loaded_revision and loaded_revision != revision:
        die(f"加载模型 revision={loaded_revision} 与请求版本 {revision} 不一致")
    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    run_prep = {**PREP, "device": device}
    if use_cuda:
        model = model.cuda()
    model.eval()

    names, seqs = read_fasta(args.fasta)
    if not names:
        die("输入 FASTA 没有序列")
    order = sorted(range(len(names)), key=lambda i: -min(len(seqs[names[i]]), MAXLEN))
    rows = []
    i = 0
    while i < len(order):
        # 组批：token 预算内（batch 内最长决定 pad）
        j = i
        while j < len(order):
            cand = order[i:j + 1]
            longest = min(len(seqs[names[cand[0]]]), MAXLEN)
            if (longest + 2) * len(cand) > args.batch_max_tokens and j > i:
                break
            j += 1
        batch = [order[k] for k in range(i, j)]
        i = j

        metas, useds = [], []
        for idx in batch:
            name = names[idx]
            full = seqs[name]
            sha = hashlib.sha256(full.encode()).hexdigest()
            ridx = resid_idx.get(name) or []
            dom_hash = hashlib.sha256(
                ",".join(map(str, ridx)).encode()).hexdigest()[:12] if ridx else "full"
            meta = {"key": "", "name": name, "model": MODEL, "revision": revision,
                    "seq_sha256": sha, "prep": run_prep, "forward_precision": PREP["forward_precision"],
                    "device": device, "len_full": len(full),
                    "len_used": min(len(full), MAXLEN), "truncated": len(full) > MAXLEN,
                    "trunc_coverage": round(min(len(full), MAXLEN) / len(full), 4),
                    "layer_set": layer_set,
                    "n_resid_stored": len(ridx) if ridx else min(len(full), MAXLEN)}
            meta["key"] = hashlib.sha256("|".join(
                [revision, sha, layer_set, prep_hash(run_prep), dom_hash]).encode()).hexdigest()[:24]
            metas.append(meta)
            useds.append(full[:MAXLEN])

        out_rows = []
        # 缓存命中检查（同序列同域共享缓存条目；meta 比对排除 name）
        todo = []
        for meta, used in zip(metas, useds):
            npz = os.path.join(args.out_dir, meta["key"] + ".npz")
            ok = False
            rejected_sha = ""
            if os.path.exists(npz):
                try:
                    with np.load(npz, allow_pickle=False) as z:
                        stored = json.loads(str(z["meta"]))
                        expected_arrays = []
                        if mean_layers:
                            expected_arrays.append("mean_layers")
                        if args.store_last_resid:
                            expected_arrays.append("resid_last")
                        if resid_layers:
                            expected_arrays.extend(["resid_layers", "resid_indices"])
                        arrays_ok = all(k in z for k in expected_arrays)
                        if arrays_ok and mean_layers:
                            arrays_ok = z["mean_layers"].ndim == 2 and z["mean_layers"].shape[0] == len(mean_layers)
                        if arrays_ok and args.store_last_resid:
                            arrays_ok = z["resid_last"].ndim == 2 and z["resid_last"].shape[0] == meta["len_used"]
                        if arrays_ok and resid_layers:
                            arrays_ok = (z["resid_layers"].ndim == 3
                                         and z["resid_layers"].shape[0] == len(resid_layers)
                                         and z["resid_layers"].shape[1] == meta["n_resid_stored"]
                                         and len(z["resid_indices"]) == meta["n_resid_stored"])
                        ok = arrays_ok and {k: v for k, v in stored.items() if k != "name"} == \
                             {k: v for k, v in meta.items() if k != "name"}
                except (OSError, KeyError, ValueError, json.JSONDecodeError):
                    ok = False
                if not ok:
                    print(f"[cache REJECT meta mismatch] {meta['name']}", file=sys.stderr)
                if ok:
                    out_rows.append({**meta, "cache": "hit", "cache_reject_sha256": ""})
                    continue
                with open(npz, "rb") as cache_file:
                    digest = hashlib.sha256(cache_file.read()).hexdigest()
                reject_dir = os.path.join(args.out_dir, "cache_rejects")
                os.makedirs(reject_dir, exist_ok=True)
                rejected_path = os.path.join(
                    reject_dir, f"{meta['key']}.{time.time_ns()}.{digest[:12]}.rejected.npz")
                shutil.move(npz, rejected_path)
                rejected_sha = digest
            todo.append((meta, used, rejected_sha))
        if todo:
            enc = tok([u for _, u, _ in todo], return_tensors="pt", padding=True,
                      truncation=True, max_length=MAXLEN + 2)
            if use_cuda:
                enc = {k: v.cuda() for k, v in enc.items()}
            t0 = time.time()
            with torch.no_grad():
                out = model(**enc)
            dt = round(time.time() - t0, 2)
            hs = out.hidden_states  # tuple(L, B, T, D), fp32 forward; fp16 only at storage
            attn = enc["attention_mask"]  # B, T
            for b, (meta, used, rejected_sha) in enumerate(todo):
                m = attn[b].bool()
                m[0] = False            # BOS
                m[int(attn[b].sum()) - 1] = False  # 末位 EOS/pad 前
                L = int(attn[b].sum()) - 2
                assert L == len(used), f"{meta['name']}: token-residue mismatch {L}!={len(used)}"
                item = {"name": meta["name"]}
                if mean_layers:
                    ml = np.stack([hs[k][b][m].mean(0).float().cpu().numpy().astype(np.float16)
                                   for k in mean_layers])  # (K, D)
                    item["mean_layers"] = ml
                if args.store_last_resid:
                    item["resid_last"] = hs[-1][b][m].float().cpu().numpy().astype(np.float16)
                if resid_layers:
                    ridx = resid_idx.get(meta["name"]) or list(range(1, L + 1))
                    assert max(ridx) <= L and min(ridx) >= 1, f"{meta['name']}: resid index 越界"
                    item["resid_layers"] = np.stack(
                        [hs[k][b][m].float().cpu().numpy()[[x - 1 for x in ridx]].astype(np.float16)
                         for k in resid_layers])  # (K, n, D)
                    item["resid_indices"] = np.array(ridx, dtype=np.int32)
                npz = os.path.join(args.out_dir, meta["key"] + ".npz")
                fd, tmp_path = tempfile.mkstemp(prefix=meta["key"] + ".", suffix=".npz.tmp",
                                                dir=args.out_dir)
                os.close(fd)
                try:
                    with open(tmp_path, "wb") as handle:
                        np.savez_compressed(handle, meta=np.array(json.dumps(meta, sort_keys=True)), **item)
                    os.replace(tmp_path, npz)
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                out_rows.append({**meta, "cache": "miss", "cache_reject_sha256": rejected_sha,
                                 "forward_seconds": dt / len(todo)})
        rows.extend(out_rows)

    man_path = os.path.join(args.out_dir, "extract_manifest.tsv")
    cols = ["key", "name", "cache", "len_full", "len_used", "truncated", "trunc_coverage",
            "seq_sha256", "revision", "layer_set", "n_resid_stored", "forward_seconds",
            "forward_precision", "device", "cache_reject_sha256"]
    if args.append_manifest and os.path.exists(man_path):
        with open(man_path, newline="") as previous:
            old_rows = list(csv.DictReader(previous, delimiter="\t"))
        run_fields = ("revision", "layer_set", "forward_precision", "device")
        for old in old_rows:
            if any(old.get(k) != rows[0].get(k) for k in run_fields):
                die(f"不可合并不同提取配置的清单: {old.get('name')}")
        combined = {r["name"]: r for r in old_rows}
        combined.update({r["name"]: r for r in rows})
        rows = [combined[name] for name in sorted(combined)]
    temp_manifest = man_path + ".tmp"
    with open(temp_manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(temp_manifest, man_path)
    print(f"[extract] samples={len(rows)} cache_hits={sum(1 for r in rows if r['cache']=='hit')} "
          f"layer_set={layer_set} -> {man_path}")
    if args.pack_matrix:
        key2idx = {}
        X = np.zeros((len(rows), len(mean_layers), 1280), dtype=np.float16)
        names_out = []
        for i, r in enumerate(rows):
            key = r["key"]
            if key not in key2idx:
                with np.load(os.path.join(args.out_dir, key + ".npz"), allow_pickle=False) as z:
                    key2idx[key] = z["mean_layers"]
            X[i] = key2idx[key]
            names_out.append(r["name"])
        np.savez_compressed(args.pack_matrix, X=X,
                            names=np.array(json.dumps(names_out)),
                            layer_set=np.array(layer_set), revision=np.array(revision))
        print(f"[extract] packed matrix {X.shape} -> {args.pack_matrix}")


if __name__ == "__main__":
    main()
