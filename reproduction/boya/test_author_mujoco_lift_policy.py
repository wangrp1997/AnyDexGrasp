import copy
import unittest

from author_mujoco_lift_policy import evaluate,penetration_summary
import author_lift as L


def fixture(pair=('/World/a','/World/b'),depth=.0015):
    rows=[]
    for i,phase in enumerate(L.POLICY['required_phases'],1):
        z=.08 if phase=='lift' else .01
        obj=dict(pose_xyzw=[0,0,z,0,0,0,1],source_bottom_m=z-.005,finger_loads_n=[1,1,0,0,0])
        rows.append(dict(step=i,time_s=i*.004,phase=phase,objects={'a':copy.deepcopy(obj),'b':copy.deepcopy(obj)},
            contacts=[dict(bodies=list(pair),separation_m=-depth)]))
    maxima=dict(joint_limit_violation_rad=0.,mimic_error_rad=0.,penetration_m=depth,palm_position_error_m=0.,
        palm_orientation_error_rad=0.,illegal_object_contact_force_n=0.,base_position_drift_m=0.,base_orientation_drift_rad=0.)
    return rows,maxima


class LiftPolicyTests(unittest.TestCase):
    def test_only_object_object_penetration_is_reclassified_without_mutating_raw_data(self):
        rows,maxima=fixture();before=copy.deepcopy(rows);m0=dict(maxima)
        result=evaluate(rows,['a','b'],maxima,6)
        self.assertFalse(result['original_gates']['penetration']);self.assertTrue(result['acceptance_gates']['penetration'])
        self.assertTrue(result['diagnostic_task_success']);self.assertEqual(rows,before);self.assertEqual(maxima,m0)

    def test_hand_ground_and_unknown_contacts_retain_original_penetration_gate(self):
        for other in ['/World/Robot/thdistal','/World/ground','/World/unknown']:
            rows,m=fixture(('/World/a',other));result=evaluate(rows,['a','b'],m,6)
            self.assertFalse(result['acceptance_gates']['penetration']);self.assertFalse(result['diagnostic_task_success'])

    def test_height_without_real_bilateral_finger_load_is_not_a_success(self):
        rows,m=fixture()
        for state in rows[-1]['objects'].values():state['finger_loads_n']=[0,1,1,0,0]
        self.assertFalse(evaluate(rows,['a','b'],m,6)['diagnostic_task_success'])

    def test_unlifted_object_does_not_become_positive_when_overlap_is_ignored(self):
        rows,m=fixture()
        for state in rows[-1]['objects'].values():state['pose_xyzw'][2]=.01;state['source_bottom_m']=.005
        self.assertFalse(evaluate(rows,['a','b'],m,6)['diagnostic_task_success'])

    def test_other_numeric_failures_and_original_deep_stop_are_not_erased(self):
        rows,m=fixture();m['mimic_error_rad']=.1
        self.assertFalse(evaluate(rows,['a','b'],m,6)['policy_admission_pass'])
        rows,m=fixture(depth=.006)
        self.assertFalse(evaluate(rows,['a','b'],m,6)['policy_admission_pass'])
        rows,m=fixture()
        self.assertFalse(evaluate(rows,['a','b'],m,6,stop='source_preclosure_stop')['policy_admission_pass'])

    def test_malformed_or_nonfinite_contacts_cannot_be_exempted(self):
        rows,_=fixture();rows[0]['contacts'][0]['separation_m']=float('nan')
        with self.assertRaises(ValueError):penetration_summary(rows,['a','b'])


if __name__=='__main__':unittest.main()
