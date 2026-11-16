"""Named commands, source gains and mimic mapping without integrating physics."""
from types import SimpleNamespace
import unittest
import numpy as np
import mujoco
import author_mujoco_preflight as ROBOT
import author_mujoco_scene_preflight as SCENE
import author_mujoco_source_filters as FILTERS
from pathlib import Path
import tempfile


def model():
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <joint name="a" range="-.2 .2"/><geom type="sphere" size=".05"/>
      <body pos="0 0 .2"><joint name="b"/><geom type="sphere" size=".04"/>
      <body pos="0 0 .2"><joint name="c"/><geom type="sphere" size=".03"/>
      </body></body></body></worldbody><actuator>
      <motor name="b" joint="b"/><motor name="a" joint="a"/>
      </actuator></mujoco>''')


class MujocoPreflightTests(unittest.TestCase):
    def test_actual_actuator_order_and_squeeze_targets_preserved(self):
        m=model();d=mujoco.MjData(m);indices=SCENE.write_controls(m,d,['a','b'],[3.,-4.])
        np.testing.assert_array_equal(indices,[1,0]);np.testing.assert_array_equal(d.ctrl,[-4.,3.])
        self.assertEqual(d.time,0.)

    def test_no_motor_can_be_omitted_or_duplicated(self):
        m=model();d=mujoco.MjData(m)
        for names,values in [(['a'],[0.]),(['a','a'],[0.,0.]),(['a','missing'],[0.,0.])]:
            with self.assertRaises(ValueError):SCENE.write_controls(m,d,names,values)

    def test_passive_mimic_is_derived_by_name(self):
        m=model();d=mujoco.MjData(m);ROBOT.put_state(m,d,['b','a'],[.3,.1],{'c':('a',1.,0.)})
        np.testing.assert_allclose(d.qpos,[.1,.3,.1]);self.assertEqual(d.time,0.)
        with self.assertRaises(ValueError):ROBOT.put_state(m,d,['a','c'],[0.,0.],{'c':('a',1.,0.)})

    def test_nonfinite_or_malformed_targets_are_rejected(self):
        m=model();d=mujoco.MjData(m)
        with self.assertRaises(ValueError):SCENE.write_controls(m,d,['a','b'],[np.nan,0.])
        with self.assertRaises(ValueError):ROBOT.put_state(m,d,['a','b'],[0.],{})

    def test_installed_and_legacy_damping_representations(self):
        for initial,expected in [(0.,.05),(np.ones(3),[.05,0.,0.])]:
            joint=SimpleNamespace(damping=initial);ROBOT.set_viscous_damping(joint,.05)
            np.testing.assert_array_equal(joint.damping,expected)
        with self.assertRaises(ValueError):ROBOT.set_viscous_damping(SimpleNamespace(damping=np.ones(2)),.05)

    def test_existing_arm_gains_and_URDF_effort_limits(self):
        names=['shoulder_pan_joint','elbow_joint','wrist_3_joint'];limits=dict(zip(names,[330.,150.,56.]))
        p=ROBOT.native_arm_parameters(names,effort_limits=limits)
        self.assertEqual([p[n]['stiffness']for n in names],[1320.,600.,216.])
        self.assertEqual([p[n]['joint_effort_limit']for n in names],[330.,150.,56.])
        with self.assertRaises(ValueError):ROBOT.native_arm_parameters(names,'unknown',effort_limits=limits)

    def test_only_explicit_source_structural_pairs_are_transferred(self):
        spec=mujoco.MjSpec.from_string('<mujoco><worldbody><body name="a"/><body name="b"/><body name="c"/></worldbody></mujoco>')
        with tempfile.TemporaryDirectory()as directory:
            path=Path(directory)/'hand.xml';path.write_text('<mujoco><contact><exclude body1="a" body2="c"/></contact></mujoco>')
            self.assertEqual(FILTERS.apply_filters(spec,path),[['a','c']]);self.assertEqual(len(spec.excludes),1)
            path.write_text('<mujoco><contact><exclude body1="a" body2="missing"/></contact></mujoco>')
            with self.assertRaises(ValueError):FILTERS.apply_filters(spec,path)


if __name__=='__main__':unittest.main()
