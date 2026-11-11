import unittest
import numpy as np
import torch
from shared_score_model import CONFIG, SharedScorer, fit_model, object_weights, predict


class SharedScoreModelTests(unittest.TestCase):
    def test_equal_object_weight(self):
        w = object_weights(["a", "a", "a", "b"])
        self.assertAlmostEqual(float(w[:3].sum()), .5)
        self.assertAlmostEqual(float(w[3]), .5)

    def test_condition_control_cannot_read_cgr(self):
        model = SharedScorer("condition_linear")
        x = np.zeros((2, 496), np.float32)
        x[1, :480] = 999
        with torch.no_grad():
            model.layers.weight.fill_(.3)
        self.assertEqual(*predict(model, x))

    def test_fit_is_deterministic_and_reload_is_exact(self):
        x = np.zeros((8, 496), np.float32)
        x[:, 0] = np.arange(8)
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        config = dict(CONFIG, epochs=8)
        a, h = fit_model(x, y, ["a"] * 4 + ["b"] * 4,
                         "shared_mlp64", 42, config)
        b, _ = fit_model(x, y, ["a"] * 4 + ["b"] * 4,
                         "shared_mlp64", 42, config)
        np.testing.assert_array_equal(predict(a, x), predict(b, x))
        fresh = SharedScorer("shared_mlp64")
        fresh.load_state_dict(a.state_dict(), strict=True)
        np.testing.assert_array_equal(predict(a, x), predict(fresh, x))
        self.assertEqual(len(h), 8)
        self.assertTrue(((predict(a, x) > 0) & (predict(a, x) < 1)).all())

    def test_nonfinite_and_wrong_input_are_rejected(self):
        with self.assertRaises(ValueError):
            fit_model(np.full((2, 496), np.nan), [0, 1], ["a", "b"],
                      "shared_linear", 1)
        with self.assertRaises(ValueError):
            SharedScorer("shared_linear")(torch.zeros(2, 480))


if __name__ == "__main__":
    unittest.main()
