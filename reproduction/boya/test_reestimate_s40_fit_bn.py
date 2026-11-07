import sys
import unittest
from pathlib import Path
import tempfile
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reestimate_s40_fit_bn as control


class FitBNTests(unittest.TestCase):
    def test_internal_excluded_and_checkpoint_order_retained(self):
        metadata = dict(trial_id=np.asarray(['a', 'b', 'c']), split=np.asarray(['train', 'internal_check', 'train']),
                        mesh_sha256=np.asarray(['m1', 'm2', 'm3']), gesture_ids=np.asarray([1, 1, 1]))
        cp = dict(fit_trial_ids=['c', 'a'], fit_mesh_sha256=['m1', 'm3'])
        self.assertEqual(control.fit_indices(metadata, cp, 1).tolist(), [2, 0])
        cp['fit_trial_ids'] = ['a', 'b', 'c']
        with self.assertRaises(ValueError):
            control.fit_indices(metadata, cp, 1)

    def test_selected_reader_never_materializes_internal_feature_rows(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'data.npz'
            features = np.ones((3, 480), np.float32)
            features[1] = np.nan
            np.savez_compressed(p, features=features, labels=np.asarray([1, 0, 1]))
            result = control.read_fit_features(p, [2, 0], 3)
            self.assertEqual(result.shape, (2, 480))
            self.assertTrue(np.isfinite(result).all())

    def test_exact_weights_unchanged_only_bn_buffers(self):
        torch.manual_seed(1)
        model = torch.nn.Sequential(torch.nn.Linear(480, 8), torch.nn.BatchNorm1d(8), torch.nn.ReLU())
        x = np.arange(4 * 480, dtype=np.float32).reshape(4, 480) / 1000
        audit = control.calibrate_model(model, x)
        self.assertTrue(audit['all_parameters_equal'])
        self.assertEqual(int(model[1].num_batches_tracked), 1)
        self.assertEqual(set(audit['changed_keys']), {'1.running_mean', '1.running_var', '1.num_batches_tracked'})
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            model[0].weight[0, 0] += 1
        with self.assertRaises(ValueError):
            control.state_change_audit(model, before)

    def test_helper_source_is_pinned(self):
        self.assertEqual(control.helper_ref()['function_source_sha256'], control.UPDATE_BN_SHA256)
        with self.assertRaises(ValueError):
            control.helper_ref('0' * 64)


if __name__ == '__main__':
    unittest.main()
