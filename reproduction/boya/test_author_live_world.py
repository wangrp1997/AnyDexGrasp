"""Live-world continuity, no-repeat ordering and real saved-plan handover."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import author_live_world_contract as K
import author_live_world as W
import audit_author_live_world as A
import author_live_world_v2 as V2
import author_live_world_v3 as V3
import author_live_world_v4 as V4

SOURCE=Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/contact_pilot_20261116_v1/scenes/00/g1_contact/pipeline')


class LiveWorldTests(unittest.TestCase):
    def test_observation_cannot_be_overwritten_by_round_reset(self):
        order=K.RoundOrder(2)
        with self.assertRaises(ValueError):order.finish()
        order.place(0);order.photograph()
        with self.assertRaises(ValueError):order.place(1)
        with self.assertRaises(ValueError):order.photograph()
        order.finish();order.place(1);order.photograph();order.finish()
        self.assertEqual(order.phase,'complete')
        with self.assertRaises(ValueError):order.place(2)

    def test_nonzero_measured_velocity_is_preserved(self):
        s=dict(sim_step=750,joint_velocity=[.001]*28,objects={'x':dict(velocity=[.0003,0,0,0,.001,0])})
        self.assertEqual(K.live_entry_errors(s,copy.deepcopy(s))['actor_pose_or_velocity_writes'],0)
        changed=copy.deepcopy(s);changed['objects']['x']['velocity']=[0]*6
        with self.assertRaises(ValueError):K.live_entry_errors(changed,s)
        changed=copy.deepcopy(s);changed['sim_step']+=1
        with self.assertRaises(ValueError):K.live_entry_errors(changed,s)

    def test_moving_entry_is_rejected_without_zeroing(self):
        for joint,body in ((.03,0),(.001,.003)):
            s=dict(sim_step=1,joint_velocity=[joint]*28,objects={'x':dict(velocity=[body,0,0,0,0,0])})
            before=copy.deepcopy(s)
            with self.assertRaises(ValueError):K.live_entry_errors(s,copy.deepcopy(s))
            self.assertEqual(s,before)

    def test_existing_world_never_starts_another_kit(self):
        world=Mock();world.launcher=object();launch=Mock(side_effect=AssertionError('A second Kit was launched'))
        result=W.World.acquire(world,launch)
        self.assertIs(result.app.world,world);launch.assert_not_called();world.bootstrap.assert_not_called()

    def test_native_handle_change_after_photograph_is_rejected(self):
        world=Mock();world.observation=dict(sim_step=750,joint_velocity=[0]*28,objects={'x':dict(velocity=[0]*6)})
        world.state.return_value=copy.deepcopy(world.observation);world.observed_handles={'camera_handle':1}
        world.capture_metadata.return_value={'camera_handle':2}
        with self.assertRaises(ValueError):W.World.validate_entry(world)

    def test_source_overlays_and_independent_audits_compile(self):
        world=Mock()
        self.assertTrue(callable(W.capture_function(world)));self.assertTrue(callable(W.live_ready_function(world)))
        self.assertTrue(callable(A.capture_auditor()));self.assertTrue(callable(A.runtime_auditor()))

    def test_fixed_camera_scope_has_no_unbound_global_in_nested_capture_code(self):
        world=Mock();world.p={'camera':{'width':1280,'height':720}}
        old=W.capture_function(world);self.assertEqual(V2.unresolved_globals(old),{'cam'})
        fixed=V2.capture_function(world);self.assertEqual(V2.unresolved_globals(fixed),set())
        self.assertIs(fixed.__globals__['cam'],world.p['camera'])

    def test_capture_native_audit_includes_primary_and_restores_original_api(self):
        audit=Mock(return_value={'hulls':893});module=SimpleNamespace(audit_scene=audit)
        def inherited(instance,scene,output):return module.audit_scene('sim','out',['all'],roots=('/World/Object',),fast=True)
        world=object.__new__(V3.World)
        with patch.dict('sys.modules',{'template_gripper_native_audit':module}),patch.object(V2.World,'capture_handles',inherited):
            self.assertEqual(world.capture_handles({},None)['hulls'],893)
        self.assertEqual(audit.call_args.args[2],['all','object']);self.assertIs(module.audit_scene,audit)

    def test_real_certified_stationary_capture_is_not_rejected_by_added_twist_cutoff(self):
        capture=Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/live_world_control_20261116_v3/scenes/00/capture')
        observed=K.B.read(capture/'LIVE_STATE.json');self.assertGreater(max(abs(np.asarray(observed['objects']['graspnet_036']['velocity'])[3:])),.02)
        proof=V4.observed_entry(copy.deepcopy(observed),observed,capture)
        self.assertEqual(proof['actor_pose_or_velocity_writes'],0);self.assertTrue(all(v['stable']for v in proof['recomputed_stationarity'].values()))
        changed=copy.deepcopy(observed);changed['objects']['graspnet_036']['velocity']=[0]*6
        with self.assertRaises(ValueError):V4.observed_entry(changed,observed,capture)

    def test_missing_original_capture_certificate_rejects_entry(self):
        with tempfile.TemporaryDirectory()as d:
            path=Path(d);K.B.save(path/'INDEPENDENT_CAPTURE_AUDIT.json',{'status':'failed'})
            K.B.save(path/'RESULTS.json',{'status':'complete'});K.B.save(path/'observed_scene.json',{'scene_id':'x'})
            s={'scene_id':'x'}
            with self.assertRaises(ValueError):V4.observed_entry(s,copy.deepcopy(s),path)

    def test_after_settling_reference_assets_follow_measurements_without_actor_writes(self):
        with tempfile.TemporaryDirectory()as d:
            out=Path(d);matrix=np.eye(4);matrix[2,3]=.0012
            scene=dict(scene_id='fixture',objects=[dict(object_id='x',object_root=d,world_T_object=matrix.tolist(),scale=1.)])
            K.B.save(out/'observed_scene.json',scene);K.B.save(out/'RESULTS.json',{'status':'complete'})
            world=Mock();world.order=K.RoundOrder(1);world.assets={'x':'stale_pre_settle_asset'};world.events=[]
            world.state.return_value=dict(scene_id='fixture',sim_step=750,objects={'x':dict(velocity=[.0003,0,0,0,0,0])})
            body=Mock();world.bank={'x':body}
            module=SimpleNamespace(object_asset=lambda args,m,scale:SimpleNamespace(pose=m.copy()))
            with patch.dict('sys.modules',{'run_template_arm_positive':module}):W.World.capture(world,0,out/'source.json',out)
            np.testing.assert_array_equal(world.assets['x'].pose,matrix)
            self.assertEqual(body.mock_calls,[]);self.assertEqual(K.B.read(out/'LIVE_STATE.json')['objects']['x']['velocity'][0],.0003)

    def observation(self):
        ready=K.B.read(SOURCE/'READY.json');scene=K.B.read(ready['scene']['path']);contract=K.M.contract_factory(K.B.read(SOURCE/'LIFT_PIPELINE_PROTOCOL.json')['inputs']['calibration']['path'])
        with np.load(ready['commands']['path'],allow_pickle=False)as z:q=dict(zip(map(str,z['joint_names']),map(float,z['initial_position'])))
        slot=contract.slots['g1:distal_000:ff'];q.update(dict(zip(contract.names,slot['independent_q'])));q.update(slot['wrist'])
        for name,(driver,mult,offset)in contract.mimics.items():q[name]=q[driver]*mult+offset
        return dict(schema=K.SCHEMA,scene_id=scene['scene_id'],sim_step=750,joint_names=list(q),joint_pos=list(q.values()),joint_velocity=[0]*len(q),
            objects={o['object_id']:dict(pose_xyzw=o['observed_pose_xyzw'],velocity=o['observed_twist'])for o in scene['objects']},
            fixture='Synthetic observation of open hand at HOME; no physics, not a captured live state')

    def test_real_saved_plan_handover_preserves_all_command_rows(self):
        with tempfile.TemporaryDirectory()as d:
            d=Path(d);observation=d/'observation.json';K.B.save(observation,self.observation())
            result=K.bind_live_plan(SOURCE,observation,d/'live');self.assertTrue(result['command_arrays_exact'])
            old=K.B.read(SOURCE/'READY.json');new=K.B.read(d/'live/READY.json')
            K.M.LIFT._verify_protocol(K.B.read(d/'live/LIFT_PIPELINE_PROTOCOL.json'))
            K.M.LIFT._verify_plan_artifacts(d/'live/native',K.B.read(new['plan']['path']))
            with np.load(old['commands']['path'],allow_pickle=False)as a,np.load(new['commands']['path'],allow_pickle=False)as b:
                for key in ('joint_target','desired_palm','phase','time_s'):np.testing.assert_array_equal(a[key],b[key])
                self.assertGreater(np.max(abs(a['initial_position']-b['initial_position'])),.1)
            self.assertEqual(K.B.read(old['candidate']['path']),K.B.read(new['candidate']['path']))

    def test_different_initial_arm_does_not_silently_reuse_old_path(self):
        with tempfile.TemporaryDirectory()as d:
            d=Path(d);s=self.observation();s['joint_pos'][s['joint_names'].index('shoulder_pan_joint')]+=.01
            K.B.save(d/'observation.json',s)
            with self.assertRaises(ValueError):K.bind_live_plan(SOURCE,d/'observation.json',d/'live')
            self.assertFalse((d/'live').exists())


if __name__=='__main__':unittest.main()
