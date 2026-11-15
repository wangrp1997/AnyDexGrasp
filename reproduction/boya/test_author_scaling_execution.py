"""CPU-only scaling entry checks; no Kit, physics or planning dispatch."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import author_scaling_execution as M
import author_gpu_capability_probe as PROBE


class Array:
    def __init__(self, value, device='cpu'):
        self.value=np.asarray(value);self.device=device
        self.shape,self.dtype=self.value.shape,self.value.dtype
    def numpy(self):return self.value


class Contact:
    max_contact_data_count=4
    sensor_count=1
    filter_count=1
    sensor_paths=['/World/envs/env_0/Object']
    filter_paths=['/World/envs/env_0/Robot/palm']
    def __init__(self,device='cpu'):self.device=device;self.calls=0
    def get_contact_data(self,dt):
        self.calls+=1
        return [Array(np.zeros((4,n),dtype=np.float32),self.device) for n in (1,3,3,1)]+[
            Array([[0]],self.device),Array([[0]],self.device)]


def sim(attrs):
    scene=types.SimpleNamespace(GetAttribute=lambda name:types.SimpleNamespace(Get=lambda:attrs[name]))
    return types.SimpleNamespace(stage=types.SimpleNamespace(GetPrimAtPath=lambda _:scene),
        cfg=types.SimpleNamespace(physics_prim_path='/World/physicsScene'),
        physics_sim_view=types.SimpleNamespace(device_ordinal=-1),
        get_physics_step_count=lambda:0,get_physics_dt=lambda:.004,
        get_setting=lambda path:False)


class ScalingTests(unittest.TestCase):
    def test_layout_is_explicit_fixed_and_never_overlapping(self):
        for count in (1,4,8):
            roots,origins=M.layout(count)
            self.assertEqual(len(roots),count)
            self.assertEqual(M.validate_layout(roots,origins,count)[0],roots)
            if count>1:
                changed=np.asarray(origins);changed[1,0]=3.99
                with self.assertRaises(ValueError):M.validate_layout(roots,changed,count)
        for count in (0,2,3,True):
            with self.assertRaises(ValueError):M.layout(count)

    def test_duplicate_template_is_validated_once_but_lane_manifests_are_distinct(self):
        calls=[]
        original=dict(source_prepared='/tmp/template',plan={'commands_sha256':'abc'},bindings={'sha':'bound'})
        jobs=M.repeated_jobs(['/tmp/template']*8,8,
            lambda paths:calls.append(paths) or [original])
        self.assertEqual(calls,[['/tmp/template']])
        self.assertEqual(len(jobs),8)
        jobs[0]['plan']['commands_sha256']='independent mutation'
        self.assertEqual(jobs[1]['plan']['commands_sha256'],'abc')
        self.assertEqual(original['plan']['commands_sha256'],'abc')
        with self.assertRaises(ValueError):M.repeated_jobs(['/tmp/a','/tmp/b'],2,lambda p:[original])

    def test_actual_backend_admission_never_falls_back(self):
        cpu={'physxScene:enableGPUDynamics':False,'physxScene:broadphaseType':'MBP',
             'physxScene:enableCCD':False,'physxScene:timeStepsPerSecond':250}
        gpu=dict(cpu,**{'physxScene:enableGPUDynamics':True,'physxScene:broadphaseType':'GPU'})
        M.backend_attributes(cpu,-1,'cpu');M.backend_attributes(gpu,0,'cuda:0')
        for attrs,ordinal,device in [(cpu,-1,'cuda:0'),(gpu,0,'cpu'),(gpu,1,'cuda:0'),
                (dict(cpu,**{'physxScene:timeStepsPerSecond':100}),-1,'cpu')]:
            with self.subTest(device=device),self.assertRaises(ValueError):M.backend_attributes(attrs,ordinal,device)

    def test_compiled_original_entries_use_dynamic_wrappers_and_admit_cpu_before_kit(self):
        vector,profile=M.compile_entries()
        self.assertIn('device',profile.__kwdefaults__)
        def stop_before_kit(*args):raise RuntimeError('validated CPU handoff before Kit')
        # This exercises the real pinned native function's first statements.
        with patch.object(M.V,'load_jobs',stop_before_kit):
            with self.assertRaisesRegex(RuntimeError,'CPU handoff'):
                vector([],Path('/unused'),env_roots=[],origins_m=[],device='cpu')
        sentinel=object()
        with patch.object(M.P,'prefix_clock',return_value=sentinel):
            self.assertIs(profile.__globals__['prefix_clock']([],None),sentinel)
        with patch.object(M.P,'sha',return_value='changed'):
            with self.assertRaisesRegex(ValueError,'source changed'):M.compile_entries()

    def test_real_probe_body_preserves_all_six_cpu_buffers_and_rejects_gpu_or_mixed_states(self):
        attrs={'physxScene:enableGPUDynamics':False,'physxScene:broadphaseType':'MBP',
               'physxScene:enableCCD':False,'physxScene:solverType':'TGS'}
        read=M.cpu_probe(PROBE)
        with tempfile.TemporaryDirectory() as tmp:
            view=Contact()
            report=read(sim(attrs),Path(tmp)/'good',contact_views={'original':(view,'filtered')})
            self.assertEqual(report['status'],'live_api_snapshot_pass')
            self.assertEqual(report['simulation_view_device_ordinal'],-1)
            self.assertEqual(view.calls,1)
            with np.load(report['snapshot']['path']) as data:self.assertEqual(len(data.files),6)
            wrong=read(sim(attrs),Path(tmp)/'gpu_contacts',contact_views={'original':(Contact('cuda:0'),'filtered')})
            self.assertEqual(wrong['status'],'failed')
            self.assertIn('Contact view failed',wrong['errors'][0])
            wrong_attrs=dict(attrs,**{'physxScene:enableGPUDynamics':True,'physxScene:broadphaseType':'GPU'})
            wrong=read(sim(wrong_attrs),Path(tmp)/'gpu_scene',contact_views={'original':(Contact(),'filtered')})
            self.assertEqual(wrong['status'],'failed')
            getter=lambda:Array([0.],'cuda:0')
            body=types.SimpleNamespace(**{name:getter for name in
                ('get_transforms','get_velocities','get_masses','get_material_properties')})
            mixed=read(sim(attrs),Path(tmp)/'mixed',contact_views={'original':(Contact(),'filtered')},rigid_views={'body':body})
            self.assertEqual(mixed['status'],'failed')
            self.assertTrue(any('CPU state tensor mismatch' in e for e in mixed['errors']))

    def test_invalid_modes_and_counts_fail_before_any_launch(self):
        for mode,count in [('cpu',4),('gpu',2),('other',1)]:
            with self.subTest(mode=mode,count=count),self.assertRaises(ValueError):
                M.run('/unused','/unused',mode=mode,lanes=count)

    def test_fetch_interruption_keeps_confirmed_lower_bound_and_attempted_upper_bound(self):
        group=dict(physical_clock_steps=100,Kit_launch_wall_s=4.,jobs=[{'observations_completed_steps':100}])
        scene=types.SimpleNamespace(sim=types.SimpleNamespace(get_physics_step_count=lambda:101))
        cost=M.physical_cost(group,scene,1)
        self.assertTrue(cost['physical_steps_unknown'])
        self.assertTrue(cost['inflight_step_unconfirmed'])
        self.assertEqual((cost['physical_steps_lower_bound'],cost['physical_steps_upper_bound']),(100,101))
        scene.sim.get_physics_step_count=lambda:100
        self.assertFalse(M.physical_cost(group,scene,1)['physical_steps_unknown'])
        self.assertTrue(M.physical_cost(group,None,1)['physical_steps_unknown'])

    def test_original_profile_writes_scaling_manifest_before_immediate_process_close(self):
        script=r'''
import json,os,sys,types
from pathlib import Path
from unittest.mock import patch
import author_scaling_execution as M
prepared,out=Path(sys.argv[1]),Path(sys.argv[2])
def save(path,value):Path(path).write_text(json.dumps(value))
storage=types.SimpleNamespace(save=save,sha=M.P.sha)
lift=types.SimpleNamespace(args_for_plan=lambda *a,**kw:None)
_,original_profile=M.compile_entries()
def vector_run(prepared_dirs,output_dir,*,device,**kw):
 assert device=='cpu' and len(prepared_dirs)==1
 scene=M.V.VectorScene([],[],[],output_dir)
 scene.sim=types.SimpleNamespace(get_physics_step_count=lambda:100)
 output_dir.mkdir()
 storage.save(output_dir/'GPU_VECTOR_EXECUTION.json',dict(clock={'prefix_limited':True,'lane_steps':[100]},
  physical_clock_steps=100,actual_integrated_environment_steps=100,recorded_command_environment_steps=100,
  complete=False,jobs=[{'label':None,'raw_label':None,'completion_tick':100,'evaluation':None}]))
 os._exit(0)
with patch.object(M,'compile_entries',return_value=(vector_run,original_profile)), \
 patch.object(M.V,'load_jobs',return_value=[{'command_steps':3024}]), \
 patch.object(M.V,'_modules',return_value=(storage,lift,None,None)), \
 patch.object(M.V,'source_bindings',return_value={}),patch.object(M.V,'verify_sources'):
 M.run(prepared,out,mode='cpu',lanes=1,prefix_ticks=100)
'''
        with tempfile.TemporaryDirectory() as tmp:
            prepared,out=Path(tmp)/'prepared',Path(tmp)/'fresh'
            prepared.mkdir()
            for name in ('commands.npz','scene.json'):(prepared/name).write_text('frozen input')
            ready={key:{'path':str(prepared/name),'sha256':M.P.sha(prepared/name)} for key,name in
                (('commands','commands.npz'),('scene','scene.json'))}
            (prepared/'READY.json').write_text(json.dumps(ready))
            env=dict(os.environ,PYTHONPATH=str(M.V.HERE)+os.pathsep+os.environ.get('PYTHONPATH',''))
            process=subprocess.run([sys.executable,'-B','-c',script,str(prepared),str(out)],env=env,text=True,capture_output=True)
            self.assertEqual(process.returncode,0,process.stderr)
            result=json.loads((out/'SCALING_EXECUTION.json').read_text())
            self.assertEqual(result['status'],'partial_scaling')
            self.assertEqual(result['mode'],'cpu')
            self.assertEqual(result['physical_clock_steps'],100)
            self.assertFalse(result['physical_steps_unknown'])
            self.assertEqual(result['actual_simulation_step_counter'],100)
            self.assertFalse(result['complete'])
            self.assertIsNone(result['raw_label'])
            self.assertEqual(result['prepared_template']['ready']['sha256'],M.P.sha(prepared/'READY.json'))
            self.assertEqual(result['group_receipt']['sha256'],M.P.sha(out/'GPU_VECTOR_EXECUTION.json'))
            self.assertEqual(result['profile']['sha256'],M.P.sha(out/'GPU_VECTOR_PROFILE.json'))


if __name__ == '__main__':unittest.main()
