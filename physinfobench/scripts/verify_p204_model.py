#!/usr/bin/env python3
"""P2.04：ESM-2 650M 本地前向验证（发现集样本、多层导出、确定性、资源实测）。

准入候选（P0.03 草案 + 用户 2026-09-24 决策委托）：本地主力=ESM-2 650M（本轮验证）；
集群档 ESM-2 3B、结构感知参考 SaProt=登记待验（HF 直连不通仅镜像可用，集群档归集群任务）。
"""
import hashlib
import json
import os
import sys
import time

import torch
from transformers import AutoModel, AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "reports/p204_model_verification.json")
MODEL = "facebook/esm2_t33_650M_UR50D"

# 发现集样本：porter_87 A 端观测序列（74 aa，dev 集，最短 strict 端点）
import csv
seq = None
B = "/Users/yuan/Documents/Codex/2026-09-08/jie/benchmark_step1"
for r in csv.DictReader(open(B + "/manifests/fold_pair_endpoint_sequence_structure_summary.tsv"), delimiter="\t"):
    if r["pdb_id"] == "2k0q" and r["requested_chain"].upper() == "A":
        seq = r["observed_sequence"]
        break
if not seq:
    sys.exit("样本序列未找到")

info = {}
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL, output_hidden_states=True, add_pooling_layer=False)
model.eval()
info["load_seconds"] = round(time.time() - t0, 1)
info["torch"] = torch.__version__
info["transformers"] = __import__("transformers").__version__
info["python"] = sys.version.split()[0]
import urllib.request, json as _json
_req = urllib.request.Request("https://hf-mirror.com/api/models/" + MODEL, headers={"User-Agent": "PhysInfoBench-P204/1.0"})
with urllib.request.urlopen(_req, timeout=30) as _r:
    _card = _json.load(_r)
info["model_card_license"] = _card.get("cardData", {}).get("license", "UNKNOWN") + "（hf-mirror API 运行时实测）"

enc = tok(seq, return_tensors="pt")
with torch.no_grad():
    t1 = time.time()
    out1 = model(**enc)
    t_fwd = time.time() - t1
    out2 = model(**enc)
hs = out1.hidden_states
info.update({
    "sample": {"pair": "porter_87_2k0qA__2lelA", "endpoint": "A", "length": len(seq)},
    "n_hidden_layers_exported": len(hs),
    "hidden_shape_last": list(hs[-1].shape),
    "forward_seconds": round(t_fwd, 2),
    "deterministic": bool(torch.equal(out1.last_hidden_state, out2.last_hidden_state)),
    "token_equivalence": int(enc["input_ids"].shape[1]) == len(seq) + 2,  # BOS/EOS
})
# 均值池化（掩码内）与指纹
mask = enc["attention_mask"].unsqueeze(-1).float()
pooled = (hs[-1] * mask).sum(1) / mask.sum(1)
info["pooled_sha256_16"] = hashlib.sha256(pooled.numpy().tobytes()).hexdigest()[:16]
info["pooled_finite"] = bool(torch.isfinite(pooled).all())
info["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None
# 逐层导出能力：中间层形状抽查
info["mid_layer_shape"] = list(hs[16].shape)
with open(OUT, "w") as f:
    json.dump(info, f, ensure_ascii=False, indent=1, sort_keys=True)
    f.write("\n")
print(json.dumps(info, ensure_ascii=False, indent=1, sort_keys=True))
