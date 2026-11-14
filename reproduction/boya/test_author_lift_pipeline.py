"""Task isolation, fixed-command preservation and fail-closed label delivery.

Heavy collection/IK/HOME/runtime stages are mocks; no Kit, physics or training.
"""
import copy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from reproduction.boya import author_lift_pipeline as P

T=P.T


def candidate(task=P.TASK):
    c=dict(schema=P.collector.CANDIDATE_SCHEMA,task=task,slot_id='g6:test:ff',
        actual_row15_camera=[0.]*15,world_T_camera=np.eye(4).tolist(),extra_depth_m=.01)
    c['candidate_sha256']=P.collector._candidate_id(c)
    return c


class Contract:
    def validate_command_arrays(self,*args,**kwargs):
        return dict(scope='Synthetic binding fixture, no physical validation')


def fixture(root):
    physics=root/'physics.json';T.save(physics,{})
    robot=root/'robot.urdf';robot.write_text('<robot name="fixture"/>')
    repair=root/'repair.json';T.save(repair,dict(schema_version=1,links={'arm':{}},
        combined_urdf_sha256=T.sha(robot),physics_config_sha256=T.sha(physics)))
    backend=root/'backend.json';T.save(backend,dict(physics_config=str(physics),robot_urdf=str(robot),
        arm_collision_refinement=str(repair),pregrasp_distance_m=.1))
    row=dict(object_id='one',object_root=str(root/'asset'),world_T_object=np.eye(4).tolist(),scale=1.)
    scene=root/'scene.json';T.save(scene,dict(schema=P.SCENE_SCHEMA,task=P.TASK,scene_id='synthetic',
        bin=None,physics_config=str(physics),target_object_id='one',objects=[row,dict(row,object_id='two')]))
    obs=root/'observation.json';T.save(obs,dict(scene_json=str(scene),scene_sha256=T.sha(scene),object_pose_as_network_input=False))
    rep=root/'rep';rep.mkdir()
    for n in ('raw_camera_candidates.npz','representation_outputs.npz','RESULTS.json'):(rep/n).write_text('mock cache')
    cal=root/'cal.json';cloud=root/'cloud.json';T.save(cal,{});T.save(cloud,{})
    return (scene,rep,obs,6,2,123,cal,cloud,backend)


def arrays(prefix=True):
    phases=['settle','home_to_pregrasp','approach']
    if not prefix:phases+=['close']*250+['hold']*50+['lift']*5
    n=len(phases)
    return dict(joint_names=np.array(['q'+str(i)for i in range(24)]),initial_position=np.zeros(24),
        phase=np.array(phases,dtype='U32'),joint_target=np.zeros((n,24)),
        desired_palm=np.repeat(np.eye(4)[None],n,axis=0),time_s=np.arange(1,n+1)*.004)


def write_plan(directory,values,**extra):
    directory.mkdir(parents=True)
    np.savez_compressed(directory/'commands.npz',**values)
    T.save(directory/'path_geometry.json',dict(ok=True,violations=[]))
    T.save(directory/'lift_arm_geometry.json',dict(ok=True,violations=[]))
    plan=dict(complete=True,admitted=False,approach_only=True,arguments=dict(pregrasp_distance_m=.04),
        commands_sha256=T.sha(directory/'commands.npz'),path_geometry_sha256=T.sha(directory/'path_geometry.json'),
        lift_arm_geometry_sha256=T.sha(directory/'lift_arm_geometry.json'),source_sha256={},extra_input_sha256={})
    plan.update(extra);T.save(directory/'plan.json',plan)
    return plan


def prepared_fixture(root):
    args=fixture(root);scene=args[0];out=root/'ready';out.mkdir()
    c=candidate();raw=root/'raw';write_plan(raw,arrays(False),admitted=True,approach_only=False,author_lift=dict(task=P.TASK))
    sources={'author_lift_pipeline.py':P.common._binding(Path(P.__file__))}
    protocol=dict(schema=P.SCHEMA,task=P.TASK,bin=None,sources=sources,
        inputs=dict(scene=P.common._binding(scene)),capture_source_bindings={},scene_archive_sources={})
    T.save(out/'LIFT_PIPELINE_PROTOCOL.json',protocol)
    refinement=P.common.stage_arm_collision_refinement(args[8],out)
    with patch.object(P,'_backend',return_value=SimpleNamespace(validate_scene=lambda p,a=None:T.read(p))),patch.object(P,'ActionContract',return_value=Contract()):
        binding=P._bind_plan(raw,out/'lift_bound',c,args[6],scene,protocol,refinement)
    T.save(out/'candidate.json',c)
    ready=dict(schema=P.SCHEMA,task=P.TASK,status='ready',protocol=P.common._binding(out/'LIFT_PIPELINE_PROTOCOL.json'),
        scene=P.common._binding(scene),candidate=P.common._binding(out/'candidate.json'),
        plan=P.common._binding(out/'lift_bound/plan.json'),commands=P.common._binding(out/'lift_bound/commands.npz'),
        command_binding=P.common._binding(out/'lift_bound/COMMAND_BINDING.json'),collection_candidate_sha256=c['candidate_sha256'])
    T.save(out/'READY.json',ready);T.save(out/'LIFT_PIPELINE_RESULTS.json',dict(status='ready',ready_sha256=T.sha(out/'READY.json')))
    return out,c,binding


def evaluation(c,binding,**changes):
    value=dict(status='complete',complete=True,label_eligible=True,total_physics_steps=23,task=P.TASK,
        completed_phases=['approach','close','hold','lift'],task_success=True,
        collection_candidate_sha256=c['candidate_sha256'],commands_sha256=binding['commands_sha256'],
        original_object_ids=['one','two'],object_outcomes=[dict(object_id='one',task_success=False),dict(object_id='two',task_success=True)])
    value.update(changes);return value


class LiftPipelineTests(unittest.TestCase):
    def test_any_original_body_label_uses_per_body_success_not_primary(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out,c,binding=prepared_fixture(root)
            p=root/'eval.json';T.save(p,evaluation(c,binding))
            result=P.collector.collect_execution(c,binding,p)
            self.assertEqual(result['label'],1);self.assertTrue(result['formal_training_eligible'])
            T.save(p,evaluation(c,binding,task_success=False,object_outcomes=[dict(object_id=x,task_success=False)for x in ['one','two']]))
            self.assertEqual(P.collector.collect_execution(c,binding,p)['label'],0)

    def test_null_infra_primary_only_and_wrong_task_never_generate_lift_labels(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out,c,binding=prepared_fixture(root);p=root/'eval.json'
            for changes in [dict(total_physics_steps=0),dict(infrastructure_error='device'),
                dict(task=P.collector.TRANSPORT_TASK),dict(completed_phases=['approach','close']),
                dict(object_outcomes=None),dict(object_outcomes=[dict(object_id='one',task_success=True)]),
                dict(object_outcomes=[dict(object_id='one',task_success=True)]*2),
                dict(object_outcomes=[dict(object_id='one',task_success=True),dict(object_id='unknown',task_success=False)]),
                dict(task_success=False),dict(object_outcomes=[dict(object_id='one',task_success=False),dict(object_id='two',task_success=1)])]:
                with self.subTest(changes=changes):
                    T.save(p,evaluation(c,binding,**changes));r=P.collector.collect_execution(c,binding,p)
                    self.assertIsNone(r['label']);self.assertFalse(r['formal_training_eligible'])

    def test_bound_lift_requires_frozen_unique_original_body_ids(self):
        c=candidate()
        with tempfile.TemporaryDirectory() as td:
            for ids in [None,[],['one','one'],[3]]:
                with self.subTest(ids=ids),self.assertRaises(ValueError):
                    P.collector.write_bound_commands(c,{},Path(td)/'bad.npz',execution_spec=dict(task=P.TASK,original_object_ids=ids))

    def test_transport_scene_and_task_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);args=fixture(root)
            with patch.object(P,'_backend',return_value=SimpleNamespace(validate_scene=lambda p,a=None:T.read(p))):
                self.assertEqual(P._scene(args[0])['bin'],None)
                for changes in [dict(schema='anydex_boya_author_scene_v1'),dict(task=P.collector.TRANSPORT_TASK),dict(bin={})]:
                    scene=T.read(args[0]);T.save(args[0],dict(scene,**changes))
                    with self.assertRaises(ValueError):P._scene(args[0])
                    T.save(args[0],scene)
            with self.assertRaises(ValueError):P._candidate(candidate(P.collector.TRANSPORT_TASK))

    def test_lift_must_preserve_every_prefix_array_and_original_close_hold_budget(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'source';dest=root/'dest'
            write_plan(source,arrays());full=arrays(False);write_plan(dest,full)
            self.assertTrue(P._verify_lift_preservation(source,dest)['source_prefix_shape_dtype_bytes_equal'])
            for key in ['joint_target','desired_palm','time_s','phase']:
                changed={k:v.copy()for k,v in full.items()}
                if key=='phase':changed[key][0]='hold'
                else:changed[key].flat[0]+=.01
                np.savez_compressed(dest/'commands.npz',**changed)
                with self.subTest(key=key),self.assertRaises(ValueError):P._verify_lift_preservation(source,dest)
            for phase in ['post_lift_hold','release']:
                changed={k:v.copy()for k,v in full.items()};changed['phase'][-1]=phase
                np.savez_compressed(dest/'commands.npz',**changed)
                with self.assertRaises(ValueError):P._verify_lift_preservation(source,dest)

    def test_rejected_direct_home_still_enters_once_driver_then_lift_without_candidate_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);args=fixture(root);out=root/'new';c=candidate();seen=[]
            backend=SimpleNamespace(validate_scene=lambda p,a=None:T.read(p))
            def collected(path,dest):
                cp=T.read(path);self.assertEqual(cp['task'],P.TASK)
                dest.mkdir();T.save(dest/'PREPARED.json',{})
                return dict(counts={'top500':500},candidates=[c,candidate()],selected_candidate_sha256=c['candidate_sha256'])
            def bridged(value,dest,**kw):
                seen.append(value['candidate_sha256']);dest.mkdir();T.save(dest/'selected_author_candidate.json',value)
            def approached(pool,dest,**kw):
                write_plan(dest/'native',arrays())
                return dict(status='no_plan',complete=True,plan_dir=str(dest/'native'))
            def routed(seed,scene,dest,**kw):
                write_plan(dest/'rrt/native',arrays(),admitted=True)
                return dict(status='ready',complete=True,admitted=True,plan_dir=str(dest/'rrt/native'),plan_sha256=T.sha(dest/'rrt/native/plan.json'))
            def lifted(seed,scene,dest,**kw):
                write_plan(dest/'native',arrays(False),admitted=True,approach_only=False,author_lift=dict(task=P.TASK))
                return dict(status='ready',complete=True,admitted=True,plan_dir=str(dest/'native'),plan_sha256=T.sha(dest/'native/plan.json'))
            sources={'author_action_semantics.py':P.common._binding(P.HERE/'author_action_semantics.py')}
            with patch.object(P,'_backend',return_value=backend),patch.object(P,'_planner',return_value=SimpleNamespace(prepare_lift=lifted)),\
                patch.object(P,'_sources',return_value=sources),patch.object(P.T,'scene_source_bindings',return_value={}),\
                patch.object(P.collector,'run_protocol',side_effect=collected),patch.object(P.bridge,'bridge_candidate',side_effect=bridged),\
                patch.object(P.common,'prepare_approach',side_effect=approached),patch.object(P,'_validate_approach'),\
                patch.object(P.home_driver,'prepare_home_route',side_effect=routed) as route,\
                patch.object(P.common,'_verify_home_preservation',return_value={'fixed':True}),patch.object(P,'ActionContract',return_value=Contract()):
                result=P.prepare(*args,out)
            self.assertEqual(result['status'],'ready');self.assertEqual(route.call_count,1)
            self.assertEqual(seen,[c['candidate_sha256']]);self.assertFalse((out/'EXECUTION_STARTED.json').exists())
            protocol=T.read(out/'LIFT_PIPELINE_PROTOCOL.json');self.assertIsNone(protocol['bin'])
            self.assertFalse(protocol['formal_collection_queue_started'])
            self.assertEqual(T.read(out/'BACKEND.json')['pregrasp_distance_m'],.04)

    def test_no_candidate_preserves_null_without_HOME_or_lift(self):
        with tempfile.TemporaryDirectory() as td:
            args=fixture(Path(td));out=Path(td)/'new'
            sources={'author_action_semantics.py':P.common._binding(P.HERE/'author_action_semantics.py')}
            with patch.object(P,'_backend',return_value=SimpleNamespace(validate_scene=lambda p,a=None:T.read(p))),\
                patch.object(P,'_sources',return_value=sources),patch.object(P.T,'scene_source_bindings',return_value={}),\
                patch.object(P.collector,'run_protocol',return_value=dict(counts={},candidates=[])),\
                patch.object(P.home_driver,'prepare_home_route') as route,patch.object(P,'_planner') as lift:
                result=P.prepare(*args,out)
            self.assertEqual(result['status'],'no_candidate');route.assert_not_called();lift.assert_not_called()

    def test_metadata_binding_preserves_all24_arrays_and_missing_old_binding(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out,c,binding=prepared_fixture(root)
            raw=P._arrays(root/'raw');bound=P._arrays(out/'lift_bound')
            for key,value in raw.items():
                self.assertEqual(value.dtype,bound[key].dtype);self.assertEqual(value.tobytes(),bound[key].tobytes())
            plan=T.read(out/'lift_bound/plan.json')
            self.assertEqual(plan['task'],P.TASK);self.assertEqual(binding['execution_spec']['original_object_ids'],['one','two'])

    def test_post_execution_hash_failure_preserves_cost_and_never_writes_label_before_close(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out,c,binding=prepared_fixture(root);closed=[]
            def run(args,scene):
                value=evaluation(c,binding);T.save(out/'execution/LIFT_EVALUATION.json',value)
                Path(scene).write_text('{}')
                return value
            def close(**kw):
                closed.append(T.read(out/'LIFT_PIPELINE_EXECUTION.json'))
                self.assertFalse((out/'COLLECTION_RECORD.json').exists())
            backend=SimpleNamespace(validate_scene=lambda p,a=None:T.read(p),args_for_plan=lambda *a:None)
            with patch.object(P,'_backend',return_value=backend),patch.object(P.common,'_launch_native',return_value=SimpleNamespace(app=SimpleNamespace(close=close))),\
                patch.object(P,'_execute_native',side_effect=run):
                with self.assertRaises(ValueError):P.execute(out)
                with self.assertRaises((ValueError,FileExistsError)):P.execute(out)
            self.assertEqual(len(closed),1);self.assertEqual(closed[0]['physics_steps'],23)
            self.assertEqual(closed[0]['status'],'infrastructure_failure');self.assertEqual(closed[0]['new_labels'],0)

    def test_geometry_artifact_tampering_prevents_launch_and_post_execution_label(self):
        for name in ['path_geometry.json','lift_arm_geometry.json']:
            for during in [False,True]:
                with self.subTest(name=name,during=during),tempfile.TemporaryDirectory() as td:
                    root=Path(td);out,c,binding=prepared_fixture(root);closed=[]
                    artifact=out/'lift_bound'/name
                    def run(args,scene):
                        value=evaluation(c,binding);T.save(out/'execution/LIFT_EVALUATION.json',value)
                        T.save(artifact,dict(ok=True,changed=True))
                        return value
                    backend=SimpleNamespace(validate_scene=lambda p,a=None:T.read(p),args_for_plan=lambda *a:None)
                    if not during:T.save(artifact,dict(ok=True,changed=True))
                    with patch.object(P,'_backend',return_value=backend),\
                        patch.object(P.common,'_launch_native',return_value=SimpleNamespace(app=SimpleNamespace(close=lambda **kw:closed.append(kw)))) as launch,\
                        patch.object(P,'_execute_native',side_effect=run):
                        with self.assertRaises(ValueError):P.execute(out)
                    self.assertFalse((out/'COLLECTION_RECORD.json').exists())
                    self.assertEqual(launch.call_count,int(during));self.assertEqual(len(closed),int(during))
                    if during:
                        receipt=T.read(out/'LIFT_PIPELINE_EXECUTION.json')
                        self.assertEqual(receipt['physics_steps'],23);self.assertEqual(receipt['new_labels'],0)
                    else:self.assertFalse((out/'EXECUTION_STARTED.json').exists())

    def test_success_record_is_durable_before_close_and_execution_cannot_repeat(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out,c,binding=prepared_fixture(root);closed=[]
            def run(args,scene):
                value=evaluation(c,binding);T.save(out/'execution/LIFT_EVALUATION.json',value);return value
            def close(**kw):
                closed.append(kw);self.assertEqual(T.read(out/'COLLECTION_RECORD.json')['label'],1)
            backend=SimpleNamespace(validate_scene=lambda p,a=None:T.read(p),args_for_plan=lambda *a:None)
            with patch.object(P,'_backend',return_value=backend),patch.object(P.common,'_launch_native',return_value=SimpleNamespace(app=SimpleNamespace(close=close))),\
                patch.object(P,'_execute_native',side_effect=run) as execute:
                self.assertEqual(P.execute(out)['label'],1)
                with self.assertRaises(FileExistsError):P.execute(out)
                self.assertEqual(execute.call_count,1)
            self.assertEqual(len(closed),1)


if __name__=='__main__':unittest.main()
