"""Regression checks for exact historical reuse, including tiny pose changes."""
from pathlib import Path
import copy
import tempfile
import unittest
import numpy as np
from run_augmented_cohort import same_pose, eligible, POSE_KEYS, CONFIG_KEYS, ROW_KEYS, EXECUTOR, EXECUTOR_SHA, save, sha


class ReuseTests(unittest.TestCase):
    def test_all_physical_fields_and_motor_order_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b'; a.mkdir();b.mkdir()
            values={k:np.arange(4,dtype=np.float64) for k in POSE_KEYS}
            values['independent_motor_names']=np.array(['thumb','finger'])
            np.savez(a/'candidates.npz',**values);np.savez(b/'candidates.npz',**values)
            self.assertTrue(same_pose(a,b))
            for key in (*POSE_KEYS,'independent_motor_names'):
                with self.subTest(key=key):
                    changed=copy.deepcopy(values)
                    if key=='independent_motor_names':changed[key]=changed[key][::-1]
                    else:changed[key][0]=np.nextafter(changed[key][0],np.inf)
                    np.savez(b/'candidates.npz',**changed)
                    self.assertFalse(same_pose(a,b))

    def test_physics_closure_and_proxy_changes_forbid_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'protocol.json';save(p,dict(source_sha256={str(EXECUTOR):EXECUTOR_SHA}))
            base={k:k for k in CONFIG_KEYS};base.update(object_row={k:k for k in ROW_KEYS},protocol_path=str(p),protocol_sha256=sha(p))
            self.assertTrue(eligible(base,base))
            mutations=[('closure_control',True),('collision_refinement','file'),('variant','refined'),('candidate_index',1)]
            for key in CONFIG_KEYS:mutations.append((key,'changed'))
            for key,value in mutations:
                with self.subTest(key=key):
                    changed=copy.deepcopy(base);changed[key]=value;self.assertFalse(eligible(changed,base))
            for key in ROW_KEYS:
                with self.subTest(row=key):
                    changed=copy.deepcopy(base);changed['object_row'][key]='changed';self.assertFalse(eligible(changed,base))


if __name__=='__main__':unittest.main()
