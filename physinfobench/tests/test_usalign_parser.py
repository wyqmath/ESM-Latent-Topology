import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from usalign_parser import parse_usalign_scores


class USAlignParserTests(unittest.TestCase):
    def test_parses_both_normalization_directions(self):
        text = ("TM-score= 0.82000 (if normalized by length of Structure_1)\n"
                "TM-score= 0.40000 (if normalized by length of Structure_2)\n")
        self.assertEqual(parse_usalign_scores(text), (.82, .4, .4))

    def test_accepts_chain_wording_in_current_output(self):
        text = ("TM-score= 0.82000 (if normalized by length of Chain_1)\n"
                "TM-score= 0.40000 (if normalized by length of Chain_2)\n")
        self.assertEqual(parse_usalign_scores(text), (.82, .4, .4))

    def test_incomplete_or_invalid_output_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Expected two"):
            parse_usalign_scores("TM-score= 0.82 (if normalized by length of Chain_1)\n")
        with self.assertRaisesRegex(ValueError, "Invalid TM"):
            parse_usalign_scores("TM-score= 1.2 (if normalized by length of Chain_1)\n"
                                 "TM-score= 0.4 (if normalized by length of Chain_2)\n")


if __name__ == "__main__":
    unittest.main()
