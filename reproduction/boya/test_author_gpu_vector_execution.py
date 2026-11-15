"""CPU-only checks of original clock reuse, frozen controls and env isolation."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
import subprocess
import json
import os
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import author_gpu_vector_execution as M


class VectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=M.CANON/'canondex/grasp/arm_parallel.py'
        spec=importlib.util.spec_from_file_location('official_native_clock_test',p)
        cls.native=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.native)

    def clock_modules(self):
        return patch.dict(sys.modules, {'canondex.grasp.arm_parallel':self.native})

    def test_explicit_layout_rejects_overlap_zoffset_and_implicit_namespace(self):
        roots=['/World/envs/env_0','/World/envs/env_1']
        self.assertEqual(M.validate_layout(roots,[[0,0,0],[4,0,0]],2)[0],roots)
        for r,o,n in [([],[],0),(roots,[[0,0,0],[3.9,0,0]],2),
                      (roots,[[0,0,0],[4,0,.1]],2),
                      (roots,[[.1,0,0],[4,0,0]],2),
                      (['/World/envs/env_0','/World/other/env_1'],[[0,0,0],[4,0,0]],2),
                      (['/World/Robot'],[[0,0,0]],1)]:
            with self.subTest(r=r,o=o,n=n),self.assertRaises(ValueError):M.validate_layout(r,o,n)

    def test_unequal_original_generators_finish_at_original_tick_without_new_command(self):
        sim=types.SimpleNamespace(tick=0);actions=[];observations=[];final=[]
        def stream(lane,length):
            yield dict(kind='ready',sim=sim,steps=length)
            for k in range(length):
                actions.append((lane,k+1,sim.tick))
                yield dict(kind='step',sim=sim,step=k+1)
                observations.append((lane,k+1,sim.tick))
            final.append((lane,sim.tick))
        def step(_):sim.tick+=1
        with self.clock_modules():result=M.shared_clock([stream(0,5),stream(1,3)],step)
        self.assertEqual(final,[(1,3),(0,5)])
        self.assertEqual(len(actions),8);self.assertEqual(len(observations),8)
        self.assertEqual(result['lane_steps'],[5,3]);self.assertEqual(result['physics_ticks'],5)
        self.assertEqual(result['recorded_environment_steps'],8)
        self.assertEqual(result['actual_integrated_environment_steps'],10)
        self.assertEqual(result['completed_lane_tail_integrated_steps'],[0,2])
        self.assertEqual(result['commands_added'],0)
        for lane,k,tick in actions:self.assertEqual(tick,k-1)
        for lane,k,tick in observations:self.assertEqual(tick,k)

    def test_native_clock_still_rejects_different_sim_and_crossed_barrier(self):
        one=object();two=object()
        def stream(sim,bad=False):
            yield dict(kind='ready',sim=sim,steps=1)
            yield dict(kind='step',sim=sim,step=2 if bad else 1)
        with self.clock_modules():
            with self.assertRaisesRegex(ValueError,'one scene'):M.shared_clock([stream(one),stream(two)],lambda _:None)
            with self.assertRaisesRegex(ValueError,'barrier'):M.shared_clock([stream(one,True)],lambda _:None)

    def test_closed_lane_keeps_its_result_and_later_failure_stays_failure(self):
        sim=types.SimpleNamespace(tick=0);final=[]
        def stream(lane,length,error=False):
            yield dict(kind='ready',sim=sim,steps=length)
            try:
                for k in range(length):
                    yield dict(kind='step',sim=sim,step=k+1)
                    if error and k==2:raise RuntimeError('real native gate/read failure')
            finally:final.append((lane,sim.tick))
        def step(_):sim.tick+=1
        with self.clock_modules(),self.assertRaisesRegex(RuntimeError,'native gate'):
            M.shared_clock([stream(0,5,True),stream(1,2)],step)
        self.assertEqual(final,[(1,2),(0,3)])

    def test_frozen_methods_change_only_namespace_and_pose_coordinates(self):
        _,lift,_,_=M._modules();methods,edits=M.adapted_methods(lift)
        self.assertEqual(edits,dict(robot_namespace=1,desired_origin=1,observed_origin=1))
        self.assertEqual(set(methods),{'ready','after_step'})
        source=Path(lift.__file__).read_text();tree=ast.parse(source)
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MultiObjectLiftRuntime')
        # Guard changes to the original frozen gates independently of adapter mocks.
        self.assertIn("POLICY['early_stop_penetration_m']",source)
        self.assertIn('common.preclosure_pushes(states, self.settled_xy, phase)',source)
        self.assertEqual(lift.POLICY['minimum_continuous_hold_s'],None)
        self.assertEqual(lift.POLICY['bilateral_force_threshold_n'],.001)
        self.assertEqual(lift.POLICY['final_reference_rise_m'],.05)
        with tempfile.TemporaryDirectory() as tmp:
            changed=Path(tmp)/'lift.py';changed.write_text(source+'\n# changed\n')
            with patch.object(lift,'__file__',str(changed)),self.assertRaisesRegex(ValueError,'Frozen lift'):
                M.adapted_methods(lift)

    def test_real_original_allbody_observation_removes_origin_and_preserves_contact_loads(self):
        import torch
        _,lift,_,_=M._modules();methods,_=M.adapted_methods(lift)
        root='/World/envs/env_1';paths={f'o{i}':root+'/AuthorScene/o'+str(i) for i in range(3)}
        robot=[root+'/Robot/thdist',root+'/Robot/ffdist']
        targets=robot+list(paths.values());count=np.zeros((3,5),np.int32);start=count.copy()
        count[1,:2]=1;start[1,:2]=[0,1]
        buffers=[np.array([[1.2],[1.4]]),np.array([[4.,.1,.06],[4.,.1,.06]]),
                 np.array([[0,0,1.],[0,0,1.]]),np.zeros((2,1)),count,start]
        bodies={}
        for i,key in enumerate(paths):
            pose=torch.tensor([[4.+.1*i,.1,.06,0,0,0,1.]])
            bodies[key]=types.SimpleNamespace(update=lambda dt:None,data=types.SimpleNamespace(
                root_link_pose_w=types.SimpleNamespace(torch=pose),root_link_vel_w=types.SimpleNamespace(torch=torch.zeros((1,6)))))
        runtime=types.SimpleNamespace(dt=.004,target_id='o0',bodies=bodies,paths=paths,contact_bodies=list(paths.values()),
            contact_targets=targets,contacts=object(),robot_contacts=types.SimpleNamespace(read=lambda:[]),
            robot_bodies=robot,legal_hand_owners={'thdist','ffdist'},max_illegal_object_force=0.,
            assets={key:types.SimpleNamespace(mesh=types.SimpleNamespace(vertices=np.array([[0.,0,0.]]))) for key in paths},
            max_penetration=0.,rows=[],contact_rows=[],settled_xy={},steps=1)
        runtime.local_pose=lambda pose:np.r_[pose[:3]-[4,0,0],pose[3:]]
        readback=types.SimpleNamespace(contact_data_to_host=lambda *a,**kw:buffers)
        with patch.dict(sys.modules,{'canondex.scene.contact_readback':readback}):methods['after_step'](runtime,'close')
        row=runtime.rows[0]
        np.testing.assert_allclose(row['objects']['o1']['pose_xyzw'][:3],[.1,.1,.06],atol=2e-7)
        self.assertAlmostEqual(row['objects']['o1']['source_bottom_m'],.06,places=6)
        self.assertEqual(row['objects']['o1']['finger_loads_n'][:2],[1.2,1.4])
        self.assertEqual(runtime.max_illegal_object_force,0.)
        self.assertEqual(len(runtime.contact_rows[0]['object_contacts']),2)
        # A nonhand contact still fails the unchanged all-body support limit.
        runtime.legal_hand_owners=set()
        with patch.dict(sys.modules,{'canondex.scene.contact_readback':readback}):methods['after_step'](runtime,'close')
        self.assertEqual(runtime.max_illegal_object_force,1.4)

    def test_source_bindings_detect_changes_without_mtime_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'source';file.write_text('frozen')
            storage,*_=M._modules();bindings={str(file):storage.sha(file)}
            M.verify_sources(bindings);file.write_text('tamper')
            with self.assertRaisesRegex(ValueError,'dependency changed'):M.verify_sources(bindings)

    def test_tail_reapplies_cached_weight_and_updates_without_changing_original_records(self):
        class Body:
            def __init__(self):self.writes=self.updates=0
            def write_data_to_sim(self):self.writes+=1
            def update(self,dt):self.updates+=1
        with tempfile.TemporaryDirectory() as tmp:
            robot=Body();bodies={f'o{i}':Body() for i in range(3)}
            class Runtime:
                def after_step(self,phase):
                    for key,body in self.bodies.items():
                        if key!=self.target_id:body.update(self.dt)
                    self.rows.append(dict(step=self.steps,phase=phase))
                    self.contact_rows.append(dict(step=self.steps,object_contacts=[dict(force=1.2)],robot_contacts=[]))
            original=Runtime();original.robot=robot;original.bodies=bodies;original.target_id='o0'
            original.dt=.004;original.output_dir=Path(tmp);original.rows=[{'original_endpoint':True}]
            original.steps=3;original.contact_rows=['original immutable contacts']
            original.max_penetration=.0002;original.max_illegal_object_force=0.
            tail=M.TailRecorder(original,{})
            for tick in (4,5):tail.before_step();tail.after_step(tick)
            result=tail.finish()
            self.assertEqual(result['steps'],2);self.assertEqual(result['body_weight_write_count'],6)
            self.assertEqual(result['new_commands'],0);self.assertEqual(robot.writes,2)
            self.assertTrue(all(b.writes==2 and b.updates==2 for b in bodies.values()))
            self.assertEqual(original.steps,3);self.assertEqual(original.rows,[{'original_endpoint':True}])
            self.assertEqual(original.contact_rows,['original immutable contacts'])
            self.assertEqual(tail.runtime.rows[-1]['shared_physics_tick'],5)
            self.assertTrue((Path(tmp)/'TAIL_CONTACTS.jsonl.gz').is_file())

    def test_failure_receipt_is_saved_before_close_which_exits_process(self):
        # Simulate the actual native fast-shutdown behavior, not a returning mock.
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'new';source=Path(tmp)/'old';source.mkdir()
            script="""
import os,sys,types,json
from pathlib import Path
from unittest.mock import patch
import author_gpu_vector_execution as M
out=Path(sys.argv[1]);source=Path(sys.argv[2])
def close(*,exit_code):
 assert (out/'GPU_VECTOR_EXECUTION.json').is_file()
 assert json.loads((out/'GPU_VECTOR_EXECUTION.json').read_text())['status']=='infrastructure_failure'
 os._exit(exit_code)
app=types.SimpleNamespace(app=types.SimpleNamespace(close=close))
modules={'isaaclab.app':types.SimpleNamespace(AppLauncher=lambda **kw:app),
 'canondex.grasp.physics':types.SimpleNamespace(load_physics_config=lambda p:None)}
job=dict(source_prepared=str(source),command_steps=3,bindings={})
with patch.dict(sys.modules,modules),patch.object(M,'load_jobs',return_value=[job]), \
 patch.object(M,'source_bindings',return_value={}), \
 patch.object(M,'bind_reference_assets',return_value={}), \
 patch.object(M,'adapted_methods',side_effect=RuntimeError('stop before native scene')):
 M.run_group([source],out,env_roots=['/World/envs/env_0'],origins_m=[[0,0,0]],device='cuda:0')
"""
            env=dict(os.environ,PYTHONPATH=str(M.HERE)+os.pathsep+os.environ.get('PYTHONPATH',''))
            process=subprocess.run([sys.executable,'-B','-c',script,str(out),str(source)],capture_output=True,text=True,env=env)
            self.assertEqual(process.returncode,1,process.stderr)
            self.assertTrue((out/'GPU_VECTOR_EXECUTION.json').is_file(),process.stderr)
            result=json.loads((out/'GPU_VECTOR_EXECUTION.json').read_text())
            self.assertEqual(result['physical_clock_steps'],0)
            self.assertEqual(result['actual_integrated_environment_steps'],0)
            self.assertIn('stop before native scene',result['error'])
            self.assertEqual(list(source.iterdir()),[])

    def test_reference_binding_precedes_real_physics_parameter_import_and_rejects_missing_scene(self):
        # Read the actual original source XML/defaults in a clean CPU process.
        script="""
import sys,json
from pathlib import Path
import author_gpu_vector_execution as M
backend=json.loads((M.HERE/'backend.collection.json').read_text())
jobs=[dict(plan=dict(arguments=backend))]
assert 'canondex.robots.hand_actuation' not in sys.modules
binding=M.bind_reference_assets(jobs)
from canondex.grasp.physics import load_physics_config
profile=load_physics_config(backend['physics_config'])
import canondex.paths as paths
import canondex.robots.hand_actuation as hand
scene=Path(backend['hand_scene'])
assert paths.HAND_SCENE==scene and hand.HAND_SCENE==scene
assert hand.hand_drive_parameters.__defaults__[1]==scene
assert hand.hand_drive_parameters(profile.hand_damping_mode)==hand.hand_drive_parameters(profile.hand_damping_mode,scene)
jobs[0]['plan']['arguments']=dict(backend,hand_scene='/missing/source/scene.xml')
try:M.bind_reference_assets(jobs)
except ValueError as e:assert 'must exist before Kit' in str(e)
else:raise AssertionError('Missing source scene was admitted')
print('actual_source_parameters_unchanged;0Kit;0physics')
"""
        env=dict(os.environ,PYTHONPATH=str(M.HERE)+os.pathsep+os.environ.get('PYTHONPATH',''))
        process=subprocess.run([sys.executable,'-B','-c',script],capture_output=True,text=True,env=env)
        self.assertEqual(process.returncode,0,process.stderr)
        self.assertIn('actual_source_parameters_unchanged',process.stdout)

    def test_clone_relationship_normalization_preserves_global_ground(self):
        self.assertEqual(M._normalize_relationship('/World/envs/env_1/Robot/th','/World/envs/env_1'),'{ENV}/Robot/th')
        self.assertEqual(M._normalize_relationship('/World/ground/CollisionPlane','/World/envs/env_1'),'/World/ground/CollisionPlane')
        self.assertNotEqual(M._normalize_relationship('/World/envs/env_0/Robot/th','/World/envs/env_1'),'{ENV}/Robot/th')


if __name__=='__main__':unittest.main()
