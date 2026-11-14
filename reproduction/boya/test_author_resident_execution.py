"""Resident boundary/lifecycle checks; these tests start no Kit or physics."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import author_resident_execution as M


class ResidentTests(unittest.TestCase):
    def assets(self, root):
        root = Path(root); args = {}; inputs = {}; objects = []
        for name in ('robot_urdf', 'hand_urdf', 'hand_scene', 'package', 'physics_config', 'diagnostic_config'):
            p = root/name; p.write_text(name); args[name] = str(p)
        for index in range(3):
            asset = root/f'o{index}'; (asset/'mesh').mkdir(parents=True); (asset/'urdf').mkdir()
            mesh = asset/'mesh/simplified.obj'; mesh.write_text(f'mesh{index}')
            urdf = asset/'urdf/coacd.urdf'; urdf.write_text(f'urdf{index}')
            hull = asset/'urdf/hull.obj'; hull.write_text(f'hull{index}')
            inputs.update({str(p): M.B.sha(p) for p in (mesh, urdf, hull)})
            pose = np.eye(4); pose[:3, 3] = [index*.1, .2, .03]
            objects.append(dict(object_id=f'o{index}', object_root=str(asset), source_mesh=str(mesh),
                source_mesh_sha256=M.B.sha(mesh), scale=1., mass_kg=.1, friction=.6,
                world_T_object=pose.tolist(), collision_refinement=None))
        return dict(task=M.B.TASK, bin=None, objects=objects, input_sha256=inputs,
            target_object_id='o0', dt_s=.004, scene_id='test'), dict(arguments=args)

    def test_full_geometry_bindings_ignore_state_but_bind_nonprimary_hulls(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene, plan = self.assets(tmp); before = M.geometry_binding(scene, plan)
            scene['objects'][2]['world_T_object'][0][3] += .1
            self.assertEqual(before, M.geometry_binding(scene, plan))
            hull = Path(tmp)/'o2/urdf/hull.obj'; hull.write_text('changed hull')
            with self.assertRaisesRegex(ValueError, 'dependency changed'):
                M.geometry_binding(scene, plan)
            scene['input_sha256'][str(hull)] = M.B.sha(hull)
            self.assertNotEqual(before['sha256'], M.geometry_binding(scene, plan)['sha256'])
            scene['objects'][1]['mass_kg'] = .2
            with self.assertRaisesRegex(ValueError, 'material'):
                M.geometry_binding(scene, plan)

    def test_native_content_aliases_preserve_physics_and_reference_checks(self):
        before = dict(inputs=dict(files={'a':'robot', 'old_repair':'repair'},
            world_T_object=np.eye(4).tolist(), dt_s=.004, base_transform=[1], mount_transform=[2]))
        after = dict(inputs=dict(before['inputs'], files={'a':'robot', 'new_repair':'repair'},
            world_T_object=(np.eye(4)*2).tolist()))
        self.assertEqual(M._native_key(before), M._native_key(after))
        after['inputs']['mount_transform'] = [3]
        self.assertNotEqual(M._native_key(before), M._native_key(after))

    def test_existing_native_reset_is_extended_to_three_bodies_without_integration(self):
        import torch
        spec = importlib.util.spec_from_file_location('native_resident_for_test', M.NATIVE_RESIDENT)
        native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
        sim = types.SimpleNamespace(steps=100, get_physics_step_count=lambda: sim.steps,
            get_physics_dt=lambda: .004)
        class Body:
            device = 'cpu'
            def __init__(self):
                self.pose = torch.tensor([[9.,9.,9.,0.,0.,0.,1.]])
                self.velocity = torch.ones((1,6)); self.resets = self.clears = 0
                self.data = types.SimpleNamespace(root_link_pose_w=types.SimpleNamespace(torch=self.pose))
                self.permanent_wrench_composer = types.SimpleNamespace(reset=self.clear)
            def clear(self): self.clears += 1
            def reset(self): self.resets += 1
            def write_root_link_pose_to_sim_index(self, *, root_pose): self.pose[:] = root_pose
            def write_root_link_velocity_to_sim_index(self, *, root_velocity): self.velocity[:] = root_velocity
            def write_data_to_sim(self): pass
            def update(self, dt): pass
        robot = Body(); robot.joint_names = ['a','b']
        robot.write_joint_state_to_sim = lambda q,v: setattr(robot, 'joint_state', (q,v))
        commands = types.SimpleNamespace(
            set_position_index=lambda *, value: setattr(robot, 'q_target', value),
            set_velocity_index=lambda *, value: setattr(robot, 'v_target', value))
        robot.actuators = types.SimpleNamespace(target_command=commands)
        bodies = {f'o{i}':Body() for i in range(3)}
        scene = dict(target_object_id='o0', objects=[dict(object_id=f'o{i}', object_root='unused',
            scale=1., world_T_object=np.eye(4).tolist()) for i in range(3)])
        for i,row in enumerate(scene['objects']): row['world_T_object'][0][3] = i*.1
        session = types.SimpleNamespace(_scene=scene, state_reset_count=0, asset_setup_count=0)
        runtime_module = types.SimpleNamespace(forward_scene=lambda _:None)
        object_asset = lambda _,T,scale: types.SimpleNamespace(pose_xyzw=np.r_[T[:3,3],[0.,0.,0.,1.]])
        with patch.dict(sys.modules, {'canondex.grasp.arm_resident_scene':native,
                'canondex.scene.runtime':runtime_module,
                'run_template_arm_positive':types.SimpleNamespace(object_asset=object_asset)}):
            controller = M._scene_type()(session)
            controller.bind(dict(sha256='first'), sim, robot, 'urdf', {'object':bodies['o0']}, np.array([0,0,0,0,0,0,1.]))
            controller.cached = dict(bodies=bodies, assets={}, paths={})
            controller.reset({'a':.2,'b':-.3}, [0.,0.,0.,0.,0.,0.,1.])
            self.assertEqual(sim.steps,100); self.assertEqual(session.state_reset_count,1)
            self.assertEqual(session.asset_setup_count,1)
            np.testing.assert_array_equal(robot.q_target.numpy(),np.array([[.2,-.3]],np.float32))
            self.assertFalse(robot.v_target.any()); self.assertFalse(robot.joint_state[1].any())
            for i,body in enumerate(bodies.values()):
                self.assertEqual(body.clears,1); self.assertFalse(body.velocity.any())
                np.testing.assert_allclose(body.pose.numpy()[0,:3], [i*.1,0,0], atol=1e-8)
            self.assertEqual(set(controller.last_resets), {'o1','o2'})
            runtime_module.forward_scene = lambda _:setattr(sim,'steps',sim.steps+1)
            with self.assertRaisesRegex(RuntimeError, 'integrated physics'):
                controller.reset({'a':.2,'b':-.3}, [0.,0.,0.,0.,0.,0.,1.])

    def test_two_jobs_use_original_stream_entry_and_one_application(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); scene,plan=self.assets(root); scene_path=root/'scene.json';M.B.save(scene_path,scene)
            plan['author_lift']={};plan_path=root/'plan.json';M.B.save(plan_path,plan)
            prepared=[]
            for i in range(2):
                d=root/f'g{i}';d.mkdir();M.B.save(d/'READY.json',dict(plan=M.B.binding(plan_path),scene=M.B.binding(scene_path)));prepared.append(d)
            closes=[];ready_calls=[];native_calls=[]
            sim=types.SimpleNamespace(get_physics_step_count=lambda:10,get_physics_dt=lambda:.004)
            class FakeRuntime:
                def __init__(self): self.bodies={}
                def factory(self):
                    self.sim=sim;self.robot='robot';self.bodies={'a':'body'};self.assets={'a':'asset'};self.paths={'a':'path'}
                    return sim,'robot','urdf',{'object':'body'}
                def ready(self): ready_calls.append(dict(self.bodies))
            class Controller:
                def __init__(self,s): self.s=s;self.runtime=None;self.cached=None;self.resets=0
                def remember(self,r): self.cached=dict(bodies=r.bodies,assets=r.assets,paths=r.paths)
                def bind(self,values): self.runtime=values;self.s.asset_setup_count+=1
                def reset(self): self.resets+=1;self.s.state_reset_count+=1
                def attach(self,r,args):
                    r.bodies=dict(self.cached['bodies']);r.sim=self.runtime[0]
                def receipt(self): return dict(resets=self.resets)
            def native_steps(args, *, resident_scene=None, runtime_factory=None):
                native_calls.append(resident_scene)
                if resident_scene.runtime is None: resident_scene.bind(runtime_factory())
                else: resident_scene.reset()
                yield {'kind':'ready'}
            native=types.SimpleNamespace(execution_steps=native_steps)
            lift=types.SimpleNamespace(__file__=str(M.E.HERE/'author_lift.py'),MultiObjectLiftRuntime=FakeRuntime)
            common=types.SimpleNamespace(_launch_native=None, transport=types.SimpleNamespace(_backend_path=lambda:None))
            pipeline=types.SimpleNamespace(common=common,_backend=lambda:lift,_scene=lambda *a:scene)
            def pipeline_execute(path):
                path=Path(path);M.B.save(path/'EXECUTION_STARTED.json', {'started':True}, exclusive=True)
                launcher=common._launch_native()
                try:
                    runtime=lift.MultiObjectLiftRuntime()
                    list(native.execution_steps(types.SimpleNamespace(), runtime_factory=runtime.factory));runtime.ready()
                    result=dict(status='complete',complete=True,label=int(path.name=='g0'),label_state='valid',physics_steps=100)
                    M.B.save(path/'LIFT_PIPELINE_EXECUTION.json',result);return result
                finally: launcher.app.close(exit_code=0)
            pipeline.execute=pipeline_execute
            def real_close(**kw):
                self.assertTrue((root/'session/SESSION_RECEIPT.json').exists());closes.append(kw)
            launcher=types.SimpleNamespace(app=types.SimpleNamespace(close=real_close))
            with patch.object(M.E,'_modules',return_value=(lift,pipeline)), \
                 patch.object(M.E,'runtime_class',return_value=(FakeRuntime,'unchanged-original-AST')), \
                 patch.object(M.E,'_launch_native_gpu0',return_value=launcher) as launch, \
                 patch.object(M,'_scene_type',return_value=Controller), \
                 patch.object(M.importlib,'import_module',return_value=native):
                session=M.ResidentExecutionSession(root/'session')
                results=[session.execute(p) for p in prepared]
                self.assertEqual([r['label'] for r in results],[1,0])
                self.assertEqual(session.kit_start_count,1);self.assertEqual(session.asset_setup_count,1)
                self.assertEqual(session.state_reset_count,1);self.assertEqual(session.deferred_close_count,2)
                self.assertEqual(len(ready_calls),2);self.assertEqual(len(native_calls),2)
                self.assertEqual(launch.call_count,1);self.assertIs(native.execution_steps,native_steps)
                self.assertFalse(closes)
                for path in prepared:
                    entry=M.B.read(path/'STREAMED_EXECUTION_ENTRY.json')
                    self.assertEqual(entry['status'],'finalized')
                    self.assertIn(str(Path(M.__file__).resolve()),entry['source_sha256'])
                session.close();session.close();self.assertEqual(len(closes),1)

    def test_changed_assets_use_official_stage_cleanup_without_new_app(self):
        with tempfile.TemporaryDirectory() as tmp:
            session=M.ResidentExecutionSession(Path(tmp)/'session');events=[]
            sim=types.SimpleNamespace(stop=lambda:events.append('stop'))
            session.controller=types.SimpleNamespace(runtime=(sim,),contacts='old',cached={},current_runtime='old')
            session._geometry={'sha256':'a'}
            fake=types.ModuleType('isaaclab.sim')
            fake.SimulationContext=types.SimpleNamespace(clear_instance=lambda:events.append('clear'))
            fake.create_new_stage=lambda:events.append('new_stage')
            package=types.ModuleType('isaaclab');package.sim=fake
            with patch.dict(sys.modules,{'isaaclab':package,'isaaclab.sim':fake}):
                session._rebuild_if_needed({'sha256':'a'});self.assertFalse(events)
                session._rebuild_if_needed({'sha256':'b'})
            self.assertEqual(events,['stop','clear','new_stage'])
            self.assertIsNone(session.controller);self.assertEqual(session.scene_rebuild_count,1)
            self.assertEqual(session.kit_start_count,0);session.close()

    def test_group_failure_stops_remaining_and_persists_before_real_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);jobs=[]
            for i in range(3): d=root/f'g{i}';d.mkdir();jobs.append(d)
            calls=[];closes=[]
            class Session:
                def __init__(self,out,**kw): self.output_dir=Path(out);self.output_dir.mkdir();self.sources={};self.poisoned=False
                def execute(self,p):
                    calls.append(p);M.B.save(Path(p)/'EXECUTION_STARTED.json',{})
                    if len(calls)==2: raise RuntimeError('Unknown interruption')
                    value=dict(label=0,label_state='valid',physics_steps=5)
                    M.B.save(Path(p)/'LIFT_PIPELINE_EXECUTION.json',value);return value
                def close(self,*,exit_code):
                    closes.append(M.B.read(self.output_dir/'EXECUTE_MANY_RESULTS.json'))
                    self.assert_code=exit_code
            with patch.object(M,'ResidentExecutionSession',Session):
                result=M.execute_many(jobs,root/'session')
            self.assertEqual(len(calls),2);self.assertEqual(result['status'],'infrastructure_failure')
            self.assertEqual(result['jobs'][2]['status'],'not_started')
            self.assertFalse(result['jobs'][2]['started']);self.assertTrue(result['execution_steps_unknown'])
            self.assertEqual(closes[0],result)

    def test_started_job_and_missing_source_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'pin';source.write_text('original')
            session=M.ResidentExecutionSession(root/'session',source_bindings={'pin':M.B.binding(source)})
            job=root/'job';job.mkdir();M.B.save(job/'EXECUTION_STARTED.json',{})
            with patch.object(M.E,'_modules',side_effect=AssertionError('must not launch')):
                with self.assertRaisesRegex(ValueError,'never restarted'):session.execute(job)
            source.unlink();receipt=session.close()
            self.assertEqual(receipt['status'],'closed_after_failure')
            self.assertIn(str(source),receipt['changed_sources'])

    def test_external_expected_pins_are_checked_before_local_name_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'frozen';source.write_text('original')
            binding=M.B.binding(source);source.write_text('tampered')
            for key in ('adapter','native_resident','official_context'):
                with self.subTest(key=key):
                    with self.assertRaisesRegex(ValueError,'dependency changed'):
                        M.ResidentExecutionSession(root/key,source_bindings={key:binding})


if __name__=='__main__': unittest.main()
