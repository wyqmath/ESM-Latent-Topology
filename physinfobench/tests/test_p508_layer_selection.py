import json
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from p508_cluster_eval import residual_layer, EXPECTED_MODEL, EXPECTED_REVISION


def cache(layers, data=None):
    meta = {"key": "abc", "layer_set": f"mean_11_23_33+resid_{'_'.join(str(x) for x in layers)}|domidx",
            "revision": EXPECTED_REVISION, "seq_sha256": "sha", "n_resid_stored": 2, "model": EXPECTED_MODEL}
    archive = {"meta": np.array(json.dumps(meta)), "resid_indices": np.array([1, 2]),
               "resid_layers": np.array([np.full((2, 3), layer) for layer in layers]) if data is None else data}
    return archive, {k: meta[k] for k in ("key", "layer_set", "revision", "seq_sha256")}


class ResidualLayerTests(unittest.TestCase):
    def test_h33_dev_slot2_external_slot0(self):
        for layers, expected_slot in [([11, 23, 33], 2), ([33], 0)]:
            archive, row = cache(layers)
            feature, meta = residual_layer(archive, row)
            np.testing.assert_array_equal(feature, np.full((2, 3), 33))
            self.assertEqual(meta["selected_cache_slot"], expected_slot)

    def test_missing_target_and_metadata_conflict_fail(self):
        archive, row = cache([11, 23])
        with self.assertRaisesRegex(ValueError, "absent"):
            residual_layer(archive, row)
        archive, row = cache([11, 23, 33])
        row["layer_set"] = "resid_33|domidx"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            residual_layer(archive, row)

    def test_array_shape_must_match_metadata(self):
        archive, row = cache([11, 23, 33], data=np.zeros((1, 2, 3)))
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            residual_layer(archive, row)

    def test_wrong_model_and_checkpoint_fail(self):
        archive, row = cache([11, 23, 33])
        metadata = json.loads(str(archive["meta"]))
        metadata["model"] = "different-model"
        archive["meta"] = np.array(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "frozen ESM-2"):
            residual_layer(archive, row)


if __name__ == "__main__":
    unittest.main()
