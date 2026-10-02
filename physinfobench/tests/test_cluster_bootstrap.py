import os
import sys
import unittest
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from cluster_bootstrap import bootstrap_auroc, component_ids, weights_for_draw, weighted_auroc


class BootstrapTest(unittest.TestCase):
    def test_hand_auroc_with_ties(self):
        # Positive 0.8 beats both negatives; positive 0.2 loses 0.4 and ties 0.2.
        self.assertAlmostEqual(weighted_auroc([1, 1, 0, 0], [.8, .2, .4, .2]), .625)

    def test_duplicate_selection_changes_auroc(self):
        labels, scores = [1, 1, 0, 0], [.8, .1, .4, .2]
        weights = weights_for_draw(['a', 'b', 'c', 'd'], ['a', 'a', 'b', 'c', 'd'])
        self.assertAlmostEqual(weighted_auroc(labels, scores), .5)
        self.assertAlmostEqual(weighted_auroc(labels, scores, weights), 2/3)
        self.assertAlmostEqual(weighted_auroc(labels, scores, weights),
                               roc_auc_score([1, 1, 1, 0, 0], [.8, .8, .1, .4, .2]))

    def test_entire_component_repeated(self):
        self.assertEqual(weights_for_draw(['g', 'g', 'h'], ['g', 'g', 'h']).tolist(), [2, 2, 1])

    def test_deterministic_paired_sampling(self):
        y, score, g = [1, 0, 1, 0], [.8, .2, .4, .6], ['a', 'a', 'b', 'b']
        a, ad, delta = bootstrap_auroc(y, score, g, B=100, seed=42, comparison_scores=score)
        b, bd, _ = bootstrap_auroc(y, score, g, B=100, seed=42)
        np.testing.assert_array_equal(ad, bd)
        np.testing.assert_array_equal(delta, np.zeros(100))
        self.assertEqual(a['n_units'], 2)
        self.assertEqual(a['ci95_low'], b['ci95_low'])

    def test_degenerate_draw_accounting(self):
        summary, draws, _ = bootstrap_auroc([1, 0], [.9, .1], ['p', 'n'], B=100, seed=13)
        self.assertEqual(summary['valid_draws'] + summary['degenerate_discarded'], 100)
        self.assertEqual(summary['degenerate_discarded'], int(np.isnan(draws).sum()))
        self.assertGreater(summary['degenerate_discarded'], 0)
        self.assertEqual(summary['ci95_low'], 1)

    def test_missing_component_fails(self):
        with self.assertRaisesRegex(ValueError, 'Missing component'):
            component_ids(['a', 'b'], {'a': 'g'})
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            component_ids(['a', 'a'], {'a': 'g'})

if __name__ == '__main__':
    unittest.main()
