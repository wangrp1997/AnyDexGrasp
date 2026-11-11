import io
import unittest
import numpy as np
import torch
from run_shared_score_control_v2 import native_metadata


class SerializationRecoveryTests(unittest.TestCase):
    def test_safe_reload_preserves_tensors_and_converts_only_metadata(self):
        tensor = torch.arange(8, dtype=torch.float32)
        source = {"model_state_dict": {"weight": tensor},
                  "fit_meshes": [np.str_("a" * 64)],
                  "config": {"epoch": np.int64(1000), "lr": np.float64(.01)}}
        converted = native_metadata(source)
        self.assertIs(converted["model_state_dict"]["weight"], tensor)
        self.assertIs(type(converted["fit_meshes"][0]), str)
        self.assertIsInstance(source["fit_meshes"][0], np.str_)
        stream = io.BytesIO()
        torch.save(converted, stream)
        stream.seek(0)
        loaded = torch.load(stream, weights_only=True)
        self.assertTrue(torch.equal(loaded["model_state_dict"]["weight"], tensor))
        self.assertEqual(loaded["config"], {"epoch": 1000, "lr": .01})


if __name__ == "__main__":
    unittest.main()
