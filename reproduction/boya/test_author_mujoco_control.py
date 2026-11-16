"""Control semantics that could silently change the fixed replay."""
import unittest
import mujoco
import numpy as np
import author_mujoco_control as C


class ControlTests(unittest.TestCase):
    def test_velocity_feedforward_sums_before_effort_clipping(self):
        spec=mujoco.MjSpec.from_string('''<mujoco><option gravity="0 0 0"/>
          <worldbody><body><joint name="a" actuatorfrcrange="-2 2"/>
          <geom type="sphere" size=".05"/></body></worldbody><actuator>
          <general name="a" joint="a" gainprm="10" biastype="affine" biasprm="0 -10 -3"
          forcelimited="true" forcerange="-2 2"/></actuator></mujoco>''')
        C.add_velocity_drives(spec,{'a':{'damping':3.}},['a'])
        m=spec.compile();d=mujoco.MjData(m)
        # Position term +10 and feedforward -9 must cancel to +1 before cap.
        C.controls(m,d,['a'],[1.],[-3.],['a']);mujoco.mj_forward(m,d)
        self.assertAlmostEqual(d.qfrc_actuator[0],1.)
        C.controls(m,d,['a'],[5.],[0.],['a']);mujoco.mj_forward(m,d)
        self.assertAlmostEqual(d.qfrc_actuator[0],2.)
        self.assertEqual(d.ctrl[0],5.)

    def test_actual_source_velocity_has_zero_hand_columns(self):
        C.T._backend_path()
        from canondex.grasp.admm.arm_only_ik import ARM_JOINTS
        names=list(ARM_JOINTS)+['hand']
        q=np.arange(35,dtype=float).reshape(5,7)/100
        velocity=C.source_velocity_targets(q,names,list(ARM_JOINTS))
        self.assertEqual(velocity.dtype,np.float32)
        np.testing.assert_array_equal(velocity[:,-1],0.)
        np.testing.assert_allclose(velocity[:,:6],17.5,atol=1e-5)

    def test_source_gates_reject_original_thresholds(self):
        maxima=dict(joint_limit_violation_rad=0.,mimic_error_rad=0.,penetration_m=0.,
            palm_position_error_m=0.,palm_orientation_error_rad=0.,illegal_object_contact_force_n=0.,
            base_position_drift_m=0.,base_orientation_drift_rad=0.)
        self.assertTrue(all(C.source_gates(maxima,10,10).values()))
        self.assertFalse(C.source_gates(dict(maxima,penetration_m=.0010001),10,10)['penetration'])
        self.assertFalse(C.source_gates(dict(maxima,mimic_error_rad=.0020001),10,10)['mimics'])
        self.assertFalse(C.source_gates(maxima,9,10)['complete_execution'])

    def test_both_contact_orders_preserve_native_provenance_and_load(self):
        rows=[dict(native_index=0,bodies=['/World/Robot/thdistal','/World/object_a'],normal_force_n=.5,separation_m=-.0001),
              dict(native_index=1,bodies=['/World/object_a','/World/Robot/ffdistal'],normal_force_n=.7,separation_m=-.0002)]
        views=C.object_views(rows,['object_a'])
        self.assertEqual([r['native_index'] for r in views],[0,1])
        loads=C.T.object_robot_loads('/World/object_a',views,['/World/Robot/thdistal','/World/Robot/ffdistal'],{'thdistal','ffdistal'})
        np.testing.assert_allclose(loads['finger_loads_n'],[.5,.7,0,0,0])
        self.assertEqual(loads['illegal_robot_support_force_n'],0.)


if __name__=='__main__':unittest.main()
