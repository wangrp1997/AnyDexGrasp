"""Outcome/error separation and real published head ingestion without updates."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import author_mujoco_sample as S
import author_mujoco_control as C
from test_author_mujoco_lift_policy import fixture


def execution(success=True):
    rows,maxima=fixture()
    return dict(status='complete',steps=6,expected_steps=6,stop=None,native_warning_count=0,
        gates=C.source_gates(maxima,6,6),maxima=maxima,evaluation=dict(task_success=success))


class SampleTests(unittest.TestCase):
    def test_completed_failure_is_an_eligible_zero_and_success_is_one(self):
        for success in (False,True):
            raw=execution(success);before=copy.deepcopy(raw)
            label=S.classify(raw)
            self.assertEqual(label['label'],int(success));self.assertTrue(label['eligible'])
            self.assertEqual(raw,before);self.assertFalse(label['original_gates']['penetration'])

    def test_unexecuted_does_not_become_a_negative(self):
        self.assertIsNone(S.classify(None)['label'])
        with tempfile.TemporaryDirectory() as d:
            r=S.write_record(d,dict(scene_id='init',split='fit'),terminal='initialization_unstable')
            self.assertIsNone(r['label']);self.assertFalse(r['eligible'])

    def test_incomplete_stop_warning_or_nonpenetration_gate_failure_remains_null(self):
        for field,value in [('steps',5),('stop','early_stop'),('native_warning_count',1),('status','physical_stop')]:
            e=execution();e[field]=value;self.assertIsNone(S.classify(e)['label'])
        for gate in set(C.LIFT.ORIGINAL_GATES)-{'penetration'}:
            e=execution();e['gates'][gate]=False;self.assertIsNone(S.classify(e)['label'])

    def test_old_deep_collision_stop_and_nonfinite_states_cannot_grant_labels(self):
        for depth in (.005001,float('nan')):
            e=execution();e['maxima']['penetration_m']=depth
            self.assertIsNone(S.classify(e)['label'])
        e=execution();del e['gates']['mimics']
        with self.assertRaises(ValueError):S.classify(e)

    def test_real_author_model_consumes_binary_samples_without_weight_updates(self):
        from author_mujoco_acceptance import dry_consumer
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'fixture.npz';out=Path(d)/'result.json'
            np.savez(path,features=np.random.default_rng(1).normal(size=(3,480)).astype(np.float32),
                labels=np.array([0,1,0],np.int64),target_index=np.array([0,19,9],np.int64),
                trial_id=np.array(['a','b','c']),mesh_sha256=np.array(['fitA','fitB','check']),
                split=np.array(['train','train','internal_check']))
            r=dry_consumer(path,out)
            self.assertTrue(r['weights_unchanged']);self.assertEqual(r['optimizer_updates'],0)
            self.assertEqual(len(r['forward_checks']),2)


if __name__=='__main__':unittest.main()
