"""Coordinate and label contracts that must hold before physical collection."""
import unittest
import numpy as np
from asset_adapter import map_slot,cgr_reference_pose
from trial_record import execution_label


class Contracts(unittest.TestCase):
    def row(self):
        return np.r_[.5,.06,.02,np.eye(3).ravel(),[.1,.2,.3]]

    def test_actual_pad_targets_and_axes(self):
        local=np.eye(4);local[:3,:3]=[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]
        local[:3,3]=[.04,.05,.06]
        palm=map_slot(self.row(),np.eye(4),{'palm_T_reference':local})
        # Independent physical endpoints: thumb/opponent lie +/-30mm on CGR Y;
        # contact reference is 20mm forward of the raw seed on CGR X.
        for sign in [-1.,1.]:
            point=local@np.r_[sign*.03,0.,0.,1.]
            np.testing.assert_allclose((palm@point)[:3],[.12,.2+sign*.03,.3],atol=1e-12)
        ref=palm@local
        np.testing.assert_allclose(ref[:3,0],[0,1,0],atol=1e-12)
        np.testing.assert_allclose(ref[:3,2],[1,0,0],atol=1e-12)

    def test_side_flip_preserves_approach(self):
        row=self.row();flipped=row.copy();flipped[3:12].reshape(3,3)[:,1:3]*=-1
        a=cgr_reference_pose(row,np.eye(4));b=cgr_reference_pose(flipped,np.eye(4))
        np.testing.assert_allclose(a[:3,2],b[:3,2])
        np.testing.assert_allclose(a[:3,0],-b[:3,0])
        np.testing.assert_allclose(a[:3,3],b[:3,3])

    def test_invalid_frames_not_repaired_into_validity(self):
        for value in [-1.,.5]:
            row=self.row();row[3]=value
            with self.assertRaises(ValueError):cgr_reference_pose(row,np.eye(4))
        for idx,value in [(1,0.),(2,-.01),(0,np.nan)]:
            row=self.row();row[idx]=value
            with self.assertRaises(ValueError):cgr_reference_pose(row,np.eye(4))

    def test_float32_roundoff_satisfies_strict_arm_transform(self):
        row=self.row();row[3]+=1e-7
        ref=cgr_reference_pose(row,np.eye(4))
        np.testing.assert_allclose(ref[:3,:3].T@ref[:3,:3],np.eye(3),atol=1e-12)

    def test_actual_attempts_required_for_binary_labels(self):
        base=dict(total_physics_steps=2100,complete=True,completed_phases=['close','lift'],label_eligible=True,task_success=False)
        self.assertEqual(execution_label(base)[0],0)
        self.assertEqual(execution_label(base|dict(task_success=True))[0],1)
        for change in [dict(total_physics_steps=0),dict(infrastructure_error=True),dict(complete=False),
                       dict(completed_phases=['approach']),dict(label_eligible=False),dict(task_success=None)]:
            self.assertIsNone(execution_label(base|change)[0])


if __name__=='__main__':unittest.main()
