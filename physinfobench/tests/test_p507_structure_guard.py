import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import p507_build_final_universe as builder


class StructureTableGuardTests(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        interim = root / "data/interim"
        isolate = interim / "p507_isolate"
        isolate.mkdir(parents=True)
        (interim / "p507_type_frozen_table.tsv").write_text(
            "chain\tstatus\tfrozen_type\nNEW_A\tfrozen\t3_1\n")
        (isolate / "seq_clusters_rep.tsv").write_text("")
        curated = root / "data/curated"
        curated.mkdir(parents=True)
        (curated / "knots.tsv").write_text("record_id\ttype_task_tier\tc2_primary\n")
        (curated / "knots_sequences.tsv").write_text("record_id\n")
        splits = root / "data/splits"
        splits.mkdir(parents=True)
        (splits / "split_manifest.tsv").write_text("task_area\tsample_id\tsplit\tdev_fold\n")
        return root, isolate

    def test_missing_or_wrong_version_tables_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self.fixture(tmp)
            with patch.object(builder, "ROOT", str(root)):
                with self.assertRaisesRegex(FileNotFoundError, "双向结构确认表"):
                    builder.main()

    def test_tm_min_must_match_both_directions_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, isolate = self.fixture(tmp)
            fields = ["query", "target", "tm_min", "tm_norm_input1", "tm_norm_input2", "returncode"]
            for name in ("usalign_edges_v2.tsv", "newnew_edges_v2.tsv"):
                with (isolate / name).open("w") as f:
                    w = csv.writer(f, delimiter="\t")
                    w.writerow(fields)
                    w.writerow(["NEW_A", "UNKNOWN", .82, .82, .40, 0])
            with patch.object(builder, "ROOT", str(root)):
                with self.assertRaisesRegex(ValueError, "tm_min 与两个方向"):
                    builder.main()
            self.assertFalse((root / "data/interim/p507_final_universe.json").exists())

    def test_complete_v2_tables_write_a_versioned_universe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, isolate = self.fixture(tmp)
            (isolate / "foldseek_hits.tsv").write_text("")
            (isolate / "nn_hits.tsv").write_text("")
            fields = "query\ttarget\ttm_min\ttm_norm_input1\ttm_norm_input2\treturncode\n"
            for name in ("usalign_edges_v2.tsv", "newnew_edges_v2.tsv"):
                (isolate / name).write_text(fields)
            with patch.object(builder, "ROOT", str(root)):
                builder.main()
            result = json.loads((root / "data/interim/p507_final_universe_20261002.json").read_text())
            self.assertIn("p507:NEW_A", result["nodes"])
            self.assertFalse((root / "data/interim/p507_final_universe.json").exists())


if __name__ == "__main__":
    unittest.main()
