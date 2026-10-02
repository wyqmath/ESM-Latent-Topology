import tempfile
import unittest
from pathlib import Path

from scripts.check_l2_stage_b_input_union import audit_union


class L2InputUnionTests(unittest.TestCase):
    def write_fixture(self, root, *, base, delta):
        manifest = root / "manifest.tsv"
        manifest.write_text(
            "uniprot_accession\tcluster_rep\tsplit\n"
            "A\tA\tdevelopment\nB\tB\tdevelopment\nC\tC\tdevelopment\n"
            "H\tH\tfinal_holdout\nEP_x\tEP_x\tdevelopment\n"
        )
        base_fa = root / "base.fa"
        base_fa.write_text("".join(f">{name}\nACD\n" for name in base))
        delta_fa = root / "delta.fa"
        delta_fa.write_text("".join(f">{name}\nEFG\n" for name in delta))
        return manifest, base_fa, delta_fa

    def test_exact_current_union_with_old_extras(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.write_fixture(Path(tmp), base=("A", "H"), delta=("B", "C"))
            audit = audit_union(*paths)
            self.assertEqual(audit["status"], "PASS")
            self.assertEqual(audit["current_development_representatives"], 3)
            self.assertEqual(audit["base_extra_historical_representatives"], 1)

    def test_missing_current_rep_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.write_fixture(Path(tmp), base=("A",), delta=("B",))
            with self.assertRaisesRegex(ValueError, "missing=1"):
                audit_union(*paths)

    def test_overlap_or_delta_extra_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.write_fixture(Path(tmp), base=("A", "B"), delta=("B", "C", "H"))
            with self.assertRaisesRegex(ValueError, "base_delta_overlap=1"):
                audit_union(*paths)


if __name__ == "__main__":
    unittest.main()
