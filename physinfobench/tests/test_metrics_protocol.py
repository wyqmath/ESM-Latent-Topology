import itertools
import sys
from pathlib import Path
import unittest

import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from metrics import average_precision, positive_rank_percentile


class MetricProtocolTests(unittest.TestCase):
    def test_average_precision_pools_ties_and_is_permutation_invariant(self):
        scores = [.9, .5, .5, .1, .1]
        labels = [1, 1, 0, 0, 1]
        expected = average_precision_score(labels, scores)
        for order in itertools.permutations(range(len(scores))):
            self.assertAlmostEqual(average_precision([scores[i] for i in order],
                                                    [labels[i] for i in order]), expected)

    def test_average_precision_matches_sklearn_for_tied_groups(self):
        rng = np.random.default_rng(13)
        for _ in range(100):
            scores = rng.integers(0, 4, size=25).astype(float)
            labels = rng.integers(0, 2, size=25)
            if labels.sum():
                self.assertAlmostEqual(average_precision(scores, labels),
                                       average_precision_score(labels, scores))

    def test_rank_percentile_uses_lower_is_better(self):
        ranked = ["best", "middle", "worst"]
        self.assertAlmostEqual(positive_rank_percentile(ranked, ["best"]), 1/3)
        self.assertAlmostEqual(positive_rank_percentile(ranked, ["worst"]), 1)


if __name__ == "__main__":
    unittest.main()
