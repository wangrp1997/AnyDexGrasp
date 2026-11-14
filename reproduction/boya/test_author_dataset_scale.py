"""Real size/split gate cases; zero model construction, updates or simulation."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import author_dataset_scale as scale
import author_training as training


def dataset(gesture, fit=1000, check=50):
    n=fit+check
    return dict(features=np.zeros((n,480),np.float32), labels=(np.arange(n)%2).astype(np.int64),
                target_index=(np.arange(n)%20).astype(np.int64),
                trial_id=np.array([f'g{gesture}_{i}' for i in range(n)]),
                mesh_sha256=np.array([f'fit_{gesture}']*fit+[f'check_{gesture}']*check),
                split=np.array(['train']*fit+['internal_check']*check),
                gesture_ids=np.full(n,gesture,np.int64))


class DatasetScaleTests(unittest.TestCase):
    def test_internal_rows_do_not_fill_fit_quota(self):
        data={g:dataset(g,fit=999,check=501) for g in scale.GESTURES}
        result=scale.assess(data)
        self.assertEqual(result['actual_total_labels'],6000)
        self.assertEqual(result['actual_fit_labels'],3996)
        self.assertEqual(result['fit_deficit'],4)
        self.assertEqual(result['status'],'insufficient_fit_labels')

    def test_full_counts_are_per_gesture_not_pooled(self):
        data={g:dataset(g,fit=1000) for g in scale.GESTURES}
        self.assertEqual(scale.assess(data)['status'],'size_ready')
        data[1]=dataset(1,fit=999);data[6]=dataset(6,fit=1001)
        result=scale.assess(data)
        self.assertEqual(result['actual_fit_labels'],4000)
        self.assertEqual(result['fit_deficit'],1)
        self.assertEqual(result['status'],'insufficient_fit_labels')

    def test_cross_gesture_trial_and_object_leakage_rejected(self):
        for mutation in ('trial','object','gesture'):
            with self.subTest(mutation=mutation):
                data={g:dataset(g,fit=100,check=5) for g in scale.GESTURES}
                if mutation=='trial': data[6]['trial_id'][0]=data[1]['trial_id'][0]
                if mutation=='object': data[6]['mesh_sha256'][-1]=data[1]['mesh_sha256'][0]
                if mutation=='gesture': data[6]['gesture_ids'][0]=1
                with self.assertRaises(ValueError): scale.assess(data,fit_per_gesture=100)

    def test_formal_training_rejects_shortfall_before_model_or_output(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(training,'components') as model:
            output=Path(tmp)/'formal'
            with self.assertRaisesRegex(ValueError,'only 999 available'):
                training.fit_head(dataset(1,fit=999,check=1001),output,gesture=1,
                                  sample_cap=1000,seed=20261114)
            model.assert_not_called()
            self.assertFalse(output.exists())


if __name__=='__main__':
    unittest.main()
