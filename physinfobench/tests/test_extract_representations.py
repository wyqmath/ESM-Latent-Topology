import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import extract_representations as extract


class FakeTokenizer:
    def __call__(self, seqs, **kwargs):
        width = max(map(len, seqs)) + 2
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, seq in enumerate(seqs):
            mask[i, :len(seq) + 2] = 1
        return {"input_ids": mask.clone(), "attention_mask": mask}


class FakeModel:
    def __init__(self):
        self.half_calls = 0
        self.cuda_calls = 0
        with open(Path(__file__).resolve().parents[1] / "configs/probes.yaml") as cfg:
            revision = extract.yaml.safe_load(cfg)["encoder"]["revision"]
        self.config = SimpleNamespace(_commit_hash=revision)

    def half(self):
        self.half_calls += 1
        return self

    def cuda(self):
        self.cuda_calls += 1
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        b, t = input_ids.shape
        hs = tuple(torch.full((b, t, 2), float(i)) for i in range(34))
        return SimpleNamespace(hidden_states=hs)


class ExtractRepresentationsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.fasta = self.root / "in.fa"
        self.fasta.write_text(">sample\nACDE\n")
        self.models = []
        self.loads = []

        def make_model(*args, **kwargs):
            self.loads.append(("model", kwargs))
            model = FakeModel()
            self.models.append(model)
            return model

        def make_tokenizer(*args, **kwargs):
            self.loads.append(("tokenizer", kwargs))
            return FakeTokenizer()

        self.model_patch = patch.object(extract.AutoModel, "from_pretrained", side_effect=make_model)
        self.tok_patch = patch.object(extract.AutoTokenizer, "from_pretrained", side_effect=make_tokenizer)
        self.threads_patch = patch.object(extract.torch, "set_num_threads")
        self.model_patch.start()
        self.tok_patch.start()
        self.threads_patch.start()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.model_patch.stop)
        self.addCleanup(self.tok_patch.stop)
        self.addCleanup(self.threads_patch.stop)

    def run_extract(self, out, *args, fasta=None):
        argv = ["extract_representations.py", "--fasta", str(fasta or self.fasta), "--out-dir", str(out), *args]
        with open(Path(__file__).resolve().parents[1] / "configs/probes.yaml") as cfg:
            revision = extract.yaml.safe_load(cfg)["encoder"]["revision"]
        with patch.object(sys, "argv", argv), patch.dict(os.environ, {"ESM_REVISION_SNAPSHOT": revision}):
            extract.main()
        with (Path(out) / "extract_manifest.tsv").open() as manifest:
            return list(csv.DictReader(manifest, delimiter="\t"))

    def test_revision_fp32_default_layers_cache_and_device_identity(self):
        out = self.root / "out"
        rows = self.run_extract(out)
        with open(Path(__file__).resolve().parents[1] / "configs/probes.yaml") as cfg:
            config = extract.yaml.safe_load(cfg)["encoder"]
        self.assertTrue(all(item[1]["revision"] == config["revision"] for item in self.loads))
        self.assertEqual(self.models[0].half_calls, 0)
        self.assertEqual(rows[0]["layer_set"], "mean_all34|domfull")
        with np.load(out / (rows[0]["key"] + ".npz"), allow_pickle=False) as z:
            self.assertEqual(z["mean_layers"].shape, (34, 2))
            meta = json.loads(str(z["meta"]))
            self.assertEqual(meta["forward_precision"], "fp32")
            self.assertEqual(meta["prep"]["device"], "cpu")
        same = self.run_extract(out)
        self.assertEqual(same[0]["cache"], "hit")
        with patch.object(extract.torch.cuda, "is_available", return_value=True), \
             patch.object(torch.Tensor, "cuda", lambda value: value):
            with self.assertRaises(SystemExit):
                self.run_extract(out, "--device", "cuda", "--append-manifest")
        self.assertEqual(self.models[-1].half_calls, 0)
        self.assertEqual(self.models[-1].cuda_calls, 1)
        with (out / "extract_manifest.tsv").open() as manifest:
            current = list(csv.DictReader(manifest, delimiter="\t"))
        self.assertEqual([row["device"] for row in current], ["cpu"])
        self.assertEqual(len(list(out.glob("*.npz"))), 2)

    def test_invalid_cache_is_quarantined_and_manifest_has_one_terminal_row(self):
        out = self.root / "out"
        rows = self.run_extract(out, "--mean-layers", "33")
        cache = out / (rows[0]["key"] + ".npz")
        with np.load(cache, allow_pickle=False) as z:
            values = {k: z[k] for k in z.files}
        meta = json.loads(str(values["meta"]))
        meta["len_used"] += 1
        values["meta"] = np.array(json.dumps(meta))
        np.savez_compressed(cache, **values)
        rejected_sha = hashlib.sha256(cache.read_bytes()).hexdigest()
        rows = self.run_extract(out, "--mean-layers", "33")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["cache"], "miss")
        self.assertEqual(rows[0]["cache_reject_sha256"], rejected_sha)
        self.assertEqual(len(list((out / "cache_rejects").glob("*.rejected.npz"))), 1)

    def test_append_manifest_keeps_disjoint_batches(self):
        out = self.root / "out"
        first = self.root / "first.fa"
        second = self.root / "second.fa"
        first.write_text(">first\nACDE\n")
        second.write_text(">second\nFGHI\n")
        self.run_extract(out, "--mean-layers", "33", "--append-manifest", fasta=first)
        rows = self.run_extract(out, "--mean-layers", "33", "--append-manifest", fasta=second)
        self.assertEqual([r["name"] for r in rows], ["first", "second"])
        self.assertEqual(len({r["device"] for r in rows}), 1)


if __name__ == "__main__":
    unittest.main()
