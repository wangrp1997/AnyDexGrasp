"""Native closure equivalence and real IK/interpolation on an analytic robot."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import author_lift_plan as lift


class AnalyticRobot:
    """Six independent pose coordinates, with the original hand/mimic layout."""
    def __init__(self):
        self.arm = list(lift._arm_names())
        fingers = ['THJ1', 'THJ2', 'THJ3', 'THJ4']+[p+j for p in ('FF','MF','RF','LF') for j in ('J2','J3','J4')]
        self.active_names = self.arm+['FAJ1','FAJ3']+fingers
        self.mimics = {p+'J1':(p+'J2', 1., 0.) for p in ('FF','MF','RF','LF')}
        self.joints = {}
        for name in self.active_names+list(self.mimics):
            joint = ET.Element('joint'); ET.SubElement(joint, 'limit', lower='-4', upper='4')
            self.joints[name] = joint
        self.lower = np.full(24,-4.); self.upper = np.full(24,4.); self.velocity = np.full(24,3.)
        self.fk = SimpleNamespace(transforms=self.transforms)

    def full_state(self, raw):
        state = dict(raw)
        for name, (driver, scale, offset) in self.mimics.items(): state[name] = state[driver]*scale+offset
        return state

    def vector(self, raw): return np.array([raw[n] for n in self.active_names])

    def transforms(self, state, base=None):
        values = [state[n] for n in self.arm]
        palm = np.eye(4); palm[:3,3] = np.array([0.,0.,.5])+values[:3]
        palm[:3,:3] = Rotation.from_euler('xyz', values[3:]).as_matrix()
        return {'palm':palm if base is None else base@palm}


class LiftPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lift.common._backend_path()
        from run_template_arm_positive import bind_hand
        from canondex.planning.model import PlanningModel
        from canondex.grasp.template_gripper_adapter import TemplateFamily
        cls.seed_dir = lift.common.ROOT/'outputs/boya_author_home_retime_20261114_v1/g1/native'
        cls.seed = lift.common.read(cls.seed_dir/'plan.json'); args = cls.seed['arguments']
        bind_hand(SimpleNamespace(hand_urdf=Path(args['hand_urdf']), hand_scene=Path(args['hand_scene'])))
        cls.model = PlanningModel(Path(args['robot_urdf']))
        cls.family = TemplateFamily.load(Path(args['package']),Path(args['hand_urdf']),Path(args['hand_scene']),cls.seed['gesture_id'])
        with np.load(cls.seed_dir/'commands.npz',allow_pickle=False) as data: cls.old={k:data[k].copy() for k in data.files}

    def test_existing_24_coordinate_prefix_is_accepted_and_not_mutated(self):
        before = {k:v.copy() for k,v in self.old.items()}
        checked = lift.validate_prefix(self.model,self.family,self.seed,self.old)
        self.assertEqual((checked['independent_coordinates'],checked['actual_coordinates'],checked['hand_dof']),(24,28,20))
        for key in before: np.testing.assert_array_equal(before[key],self.old[key])

    def test_transport_closed_or_changed_open_prefix_is_rejected(self):
        for mutation in ('transport','closed','hand','timestep','palm','nonfinite'):
            with self.subTest(mutation=mutation):
                seed=copy.deepcopy(self.seed); arrays={k:v.copy() for k,v in self.old.items()}
                if mutation=='transport':seed['author_transport']={'task':'old bin'}
                if mutation=='closed':arrays['phase'][-1]='close'
                if mutation=='hand':arrays['joint_target'][0,-1]+=.01
                if mutation=='timestep':arrays['time_s'][-1]+=.004
                if mutation=='palm':arrays['desired_palm'][-1,0,3]+=.01
                if mutation=='nonfinite':arrays['joint_target'][0,0]=np.nan
                with self.assertRaises(ValueError):lift.validate_prefix(self.model,self.family,seed,arrays)

    def test_original_250_close_50_hold_all_24_commands_are_exact(self):
        commands=list(self.old['joint_target'].copy()); phases=self.old['phase'].tolist(); desired=list(self.old['desired_palm'].copy())
        reference_commands=copy.deepcopy(commands); reference_phases=list(phases); reference_desired=copy.deepcopy(desired)
        last,audit=lift.append_source_closure(self.model,self.family,self.seed,commands,phases,desired)
        original_audit=lift.common._append_source_closure(self.model,self.family,self.seed,reference_commands,reference_phases,reference_desired)
        np.testing.assert_array_equal(commands,reference_commands)
        np.testing.assert_array_equal(phases,reference_phases)
        np.testing.assert_array_equal(desired,reference_desired)
        self.assertEqual(audit,original_audit)
        self.assertEqual(phases[-300:],['close']*250+['hold']*50)
        np.testing.assert_array_equal([last[n] for n in self.family.hand.names],self.seed['closed_motors'])
        fixed=[i for i,n in enumerate(self.model.active_names) if n in list(lift._arm_names())+['FAJ1','FAJ3']]
        np.testing.assert_array_equal(np.asarray(commands)[-300:,fixed],np.broadcast_to(self.old['joint_target'][-1,fixed],(300,8)))

    def analytic_fixture(self):
        model=AnalyticRobot(); raw={n:0. for n in model.active_names}
        for name in model.active_names[6:]:raw[name]=.12
        last=model.full_state(raw); palm=model.fk.transforms(last)['palm']
        reference=SimpleNamespace(arm_home={n:0. for n in model.arm},base_transform=np.eye(4))
        return model,reference,{'palm':palm.tolist()},last

    def test_real_native_32_ik_lift_retains_hand_and_reaches_exact_world_z_target(self):
        model,reference,seed,last=self.analytic_fixture()
        commands=[];phases=[];desired=[];progress=[]
        timing,error=lift.append_source_lift(model,reference,seed,last,commands,phases,desired,progress)
        self.assertEqual(len(progress),32);self.assertTrue(all(p['completed'] for p in progress))
        self.assertTrue(all(p['nfev']<=300 for p in progress))
        self.assertGreaterEqual(timing['duration_s'],2.)
        self.assertLessEqual(error['position_m'],1e-4)
        expected=np.asarray(seed['palm']);expected[2,3]+=.1
        np.testing.assert_array_equal(desired[-1],expected)
        np.testing.assert_allclose(model.fk.transforms(model.full_state(dict(zip(model.active_names,commands[-1]))))['palm'],expected,atol=1e-10)
        fixed=[i for i,n in enumerate(model.active_names) if n not in model.arm]
        np.testing.assert_array_equal(np.asarray(commands)[:,fixed],np.broadcast_to(model.vector(last)[fixed],(len(commands),18)))
        self.assertEqual(set(phases),{'lift'})
        self.assertEqual(lift.POLICY['additional_lift_hold_steps'],0)
        audit=lift.home.derivative_audit(np.asarray(commands)[:,:6],model.vector(last)[:6],model.velocity[:6],model.lower[:6],model.upper[:6])
        self.assertTrue(audit['ok'])

    def test_local_ik_failure_is_one_attempt_without_alternative_seed(self):
        model,reference,seed,last=self.analytic_fixture();progress=[]
        import canondex.grasp.admm.arm_only_ik as native
        with patch.object(native,'arm_only_palm_ik',side_effect=RuntimeError('original budget exhausted')) as solver:
            with self.assertRaises(lift.LocalPlanningRejected):
                lift.append_source_lift(model,reference,seed,last,[],[],[],progress)
        self.assertEqual(solver.call_count,1);self.assertEqual(len(progress),1)
        self.assertFalse(progress[0]['completed'])

    def test_broken_ik_input_is_not_reported_as_geometric_null(self):
        model,reference,seed,last=self.analytic_fixture()
        import canondex.grasp.admm.arm_only_ik as native
        with patch.object(native,'arm_only_palm_ik',side_effect=ValueError('broken input')):
            with self.assertRaises(ValueError):
                lift.append_source_lift(model,reference,seed,last,[],[],[],[])


if __name__=='__main__':unittest.main()
