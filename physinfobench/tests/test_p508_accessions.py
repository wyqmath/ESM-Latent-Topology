import sys
from pathlib import Path
import unittest
import gemmi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from p508_fix_exclusions import canonical_accession, resolve_accessions


def block(refs):
    return gemmi.cif.read_string("data_test\nloop_\n_struct_asym.id\n_struct_asym.entity_id\nA 1\nB 2\n"
                               "loop_\n_struct_ref.entity_id\n_struct_ref.db_name\n_struct_ref.pdbx_db_accession\n" + refs).sole_block()


class AccessionTests(unittest.TestCase):
    def test_exact_chain_entity_not_last_entry(self):
        b = block("1 UNP P55408\n2 UNP P55407\n")
        self.assertEqual(resolve_accessions(b, "A")["accessions"], ["P55408"])
        self.assertEqual(resolve_accessions(b, "B")["accessions"], ["P55407"])

    def test_db_code_prefix_not_accession(self):
        self.assertIsNone(canonical_accession("TRAM_RHISN"))
        self.assertIsNone(canonical_accession("TRAM"))
        self.assertEqual(canonical_accession("P55408-2"), "P55408")

    def test_unknown_and_ambiguous_explicit(self):
        self.assertEqual(resolve_accessions(block("1 PDB 2Q2C\n"), "A")["status"], "no_uniprot_reference")
        self.assertEqual(resolve_accessions(block("1 UNP P55408\n1 UNP P55407\n"), "A")["status"], "ambiguous_multi_accession")
        self.assertEqual(resolve_accessions(block("1 UNP ?\n"), "A")["status"], "invalid_accession")
        self.assertEqual(resolve_accessions(block("1 UNP P55408\n"), "C")["status"], "missing_or_ambiguous_asym_entity")


if __name__ == "__main__":
    unittest.main()
