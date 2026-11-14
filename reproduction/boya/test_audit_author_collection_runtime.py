"""Complete synthetic production-shaped records; no planner or native startup."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from reproduction.boya import audit_author_collection_runtime as audit
from reproduction.boya import author_collection_execute as stream
from reproduction.boya import author_lift as lift


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def bind(path):return dict(path=str(Path(path).resolve()),sha256=audit.sha(path))


class Fixture:
    def __init__(self,root,success=True,compressed=True):
        self.root=Path(root);self.trial=self.root/'trial';self.trial.mkdir();self.execution=self.trial/'execution';self.execution.mkdir()
        self.ids=['primary','other','third'];self.objects=[]
        for i,key in enumerate(self.ids):
            mesh=self.root/key/'mesh/simplified.obj';mesh.parent.mkdir(parents=True)
            # A different source PLY identity is represented by a valid tiny
            # ASCII fixture, kept separate from actual execution OBJ geometry.
            visual=self.root/key/'nontextured.ply';visual.write_text('original visual '+key)
            mesh.write_text('v -.01 -.01 -.01\nv .01 -.01 -.01\nv 0 .01 -.01\nv 0 0 .01\nf 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n')
            pose=np.eye(4);pose[:3,3]=[i*.05,0,.011]
            self.objects.append(dict(object_id=key,object_root=str(mesh.parent.parent),source_mesh=str(mesh),
                source_mesh_sha256=audit.sha(mesh),source_visual_mesh=str(visual),source_visual_mesh_sha256=audit.sha(visual),
                mesh_group_sha256=audit.sha(visual),world_T_object=pose.tolist(),scale=1.,mass_kg=.1,friction=.6))
        self.scene=self.root/'scene.json';save(self.scene,dict(schema=lift.SCENE_SCHEMA,task=lift.TASK,bin=None,scene_id='synthetic',
            dt_s=.004,objects=self.objects,original_object_ids=self.ids,target_object_id='primary',dataset_split='fit',
            all_source_mesh_sha256=[o['source_visual_mesh_sha256'] for o in self.objects],source_sha256={},input_sha256={}))
        self.split=self.root/'split.json';save(self.split,dict(internal_mesh_groups=dict(fit=[dict(object_id=o['object_id'],mesh_sha256=o['source_visual_mesh_sha256'])for o in self.objects],internal_check=[])))
        self.robot=self.root/'robot.urdf';self.hand=self.root/'hand.urdf'
        self.names=[f'arm{i}'for i in range(6)]+['FAJ1','FAJ3']+[f'motor{i}'for i in range(16)]
        links=[f'armbody{i}'for i in range(6)]+['wrist','pmbase']+['thbody0','ffbody1']+[f'fingerbody{i}'for i in range(2,16)]
        joint_xml=''.join(f'<joint name="{name}" type="revolute"><parent link="base"/><child link="{link}"/><limit lower="-2" upper="2"/></joint>'for name,link in zip(self.names,links))
        mimics=''.join(f'<joint name="passive{i}" type="revolute"><parent link="palm"/><child link="passivebody{i}"/><limit lower="-2" upper="2"/><mimic joint="motor{i}" multiplier="1" offset="0"/></joint>'for i in range(4))
        self.robot.write_text('<robot>'+joint_xml+mimics+'<joint name="PALM" type="fixed"><parent link="pmbase"/><child link="palm"/></joint></robot>')
        self.hand.write_text('<robot>'+''.join(f'<joint name="m{i}" type="revolute"><parent link="palm"/><child link="{name}"/></joint>'for i,name in enumerate(links[8:]+[f'passivebody{i}'for i in range(4)]))+'</robot>')
        self.bound=self.trial/'lift_bound';self.bound.mkdir()
        self.phases=np.array(sum(([p]*n for p,n in zip(audit.PHASES,[50,1,500,250,50,500])),[]))
        self.n=len(self.phases);targets=np.broadcast_to(np.linspace(-.1,.1,24),(self.n,24)).copy()
        self.arrays=dict(joint_names=np.array(self.names),joint_target=targets,initial_position=targets[0],phase=self.phases,
            desired_palm=np.broadcast_to(np.eye(4),(self.n,4,4)).copy(),time_s=np.arange(1,self.n+1)*.004)
        canonical=np.arange(480,dtype=np.float32)
        self.candidate=dict(task=lift.TASK,canonical_cgr480=canonical.tolist(),canonical_cgr480_identity=dict(c_order_bytes_sha256=__import__('hashlib').sha256(canonical.tobytes()).hexdigest()),
            canonicalization=dict(applied_once=True),raw_depth_index=2,extra_depth_index=1)
        self.candidate['candidate_sha256']=audit.json_sha(self.candidate)
        self.candidate_path=self.trial/'candidate.json';save(self.candidate_path,self.candidate)
        spec=dict(task=lift.TASK,original_object_ids=self.ids)
        np.savez_compressed(self.bound/'commands.npz',**self.arrays,collection_candidate_sha256=np.array(self.candidate['candidate_sha256']),collection_execution_spec_sha256=np.array(audit.json_sha(spec)))
        self.command_binding=dict(candidate_sha256=self.candidate['candidate_sha256'],commands_path=str(self.bound/'commands.npz'),commands_sha256=audit.sha(self.bound/'commands.npz'),
            execution_spec=spec,action_contract_validated=True)
        self.command_binding['command_binding_sha256']=audit.json_sha(self.command_binding)
        save(self.bound/'COMMAND_BINDING.json',self.command_binding)
        for name in ('path_geometry.json','lift_arm_geometry.json'):save(self.bound/name,dict(ok=True))
        self.plan=dict(task=lift.TASK,complete=True,admitted=True,approach_only=False,arguments=dict(robot_urdf=str(self.robot),hand_urdf=str(self.hand),pregrasp_distance_m=.04),
            commands_sha256=self.command_binding['commands_sha256'],path_geometry_sha256=audit.sha(self.bound/'path_geometry.json'),lift_arm_geometry_sha256=audit.sha(self.bound/'lift_arm_geometry.json'),
            author_collection_binding=dict(collection_candidate_sha256=self.candidate['candidate_sha256']),source_sha256={},input_sha256={},extra_input_sha256={})
        save(self.bound/'plan.json',self.plan)
        self.protocol=dict(task=lift.TASK,bin=None,inputs=dict(scene=bind(self.scene)),sources={},capture_source_bindings={},original_object_ids=self.ids,
            primary_is_candidate_target=False,candidate_budget=1,fallback_candidate=False)
        save(self.trial/'LIFT_PIPELINE_PROTOCOL.json',self.protocol)
        self.ready=dict(task=lift.TASK,scene=bind(self.scene),protocol=bind(self.trial/'LIFT_PIPELINE_PROTOCOL.json'),candidate=bind(self.candidate_path),plan=bind(self.bound/'plan.json'),
            commands=bind(self.bound/'commands.npz'),command_binding=bind(self.bound/'COMMAND_BINDING.json'),collection_candidate_sha256=self.candidate['candidate_sha256'])
        save(self.trial/'READY.json',self.ready)
        save(self.trial/'LIFT_PIPELINE_RESULTS.json',dict(status='ready',ready_sha256=audit.sha(self.trial/'READY.json')))
        save(self.trial/'EXECUTION_STARTED.json',dict(task=lift.TASK,ready_sha256=audit.sha(self.trial/'READY.json')))
        paths={key:'/World/Objects/'+key for key in self.ids};robot=['/World/Robot/thbody0','/World/Robot/ffbody1','/World/Robot/armbody0']
        joints,legal=audit._robot_xml(self.plan,audit.Bindings())
        setup=dict(body_paths=paths,contact_targets=robot+list(paths.values())+['/World/Ground'],physical_state_writes_during_execution=0,
            global_gravity_m_s2=[0.,0.,0.],every_dynamic_body_weight_n=[0.,0.,-.98],native=dict(within_20um=True),legal_hand_rigid_owners=sorted(legal),
            entry={key:dict(pose_xyzw=[i*.05,0,.011,0,0,0,1],velocity=[0.]*6)for i,key in enumerate(self.ids)})
        save(self.execution/'MULTI_OBJECT_SETUP.json',setup)
        self.rows=[];self.contacts=[];lift_i=0
        for i,phase in enumerate(self.phases):
            if phase=='lift':lift_i+=1
            objects={};contact=[]
            for j,key in enumerate(self.ids):
                dz=.06*lift_i/500 if key=='other'and success else 0.
                loaded=key=='other'and phase in ('close','hold','lift') and success
                fingers=[.01,.012,0,0,0] if loaded else [0.]*5
                objects[key]=dict(pose_xyzw=[j*.05,0,.011+dz,0,0,0,1],velocity=[0.]*6,source_bottom_m=.001+dz,
                    finger_loads_n=fingers,robot_load_n=sum(fingers),illegal_robot_support_force_n=0.)
                if loaded:
                    for body,force in zip(robot[:2],fingers):contact.append(dict(bodies=[paths[key],body],normal_force_n=force,separation_m=-1e-7,position_world_m=[0,0,.02],raw_normal_world=[0,0,1]))
            self.rows.append(dict(step=i+1,time_s=(i+1)*.004,phase=str(phase),objects=objects))
            self.contacts.append(dict(step=i+1,object_contacts=contact,robot_contacts=[dict(bodies=[robot[2],'/World/Ground'],normal_force_n=0.,separation_m=-1e-7,position_world_m=[0,0,0])]))
        save(self.execution/'MULTI_OBJECT_TRACE.json',dict(rows=self.rows))
        if compressed:
            journal=stream.ContactJournal(self.execution/'MULTI_OBJECT_CONTACTS.jsonl.gz')
            for row in self.contacts:journal.append(row)
            save(self.execution/'MULTI_OBJECT_CONTACTS.json',journal.pointer())
        else:save(self.execution/'MULTI_OBJECT_CONTACTS.json',dict(rows=self.contacts))
        actual_names=list(reversed(list(joints)))
        actual_targets=audit._full_target(self.names,self.arrays['joint_target'],joints,actual_names)
        self.native_arrays=dict(time_s=self.arrays['time_s'],phase=self.phases,desired_palm=self.arrays['desired_palm'],joint_names=np.array(actual_names),
            joint_target=actual_targets,joint_pos=actual_targets.copy(),object_pose_xyzw=np.array([r['objects']['primary']['pose_xyzw']for r in self.rows]))
        np.savez_compressed(self.execution/'trace.npz',**self.native_arrays)
        save(self.execution/'evaluation.json',dict(complete=True,total_physics_steps=self.n,gates=dict.fromkeys(audit.GATES[:7],True),trace_sha256=audit.sha(self.execution/'trace.npz')))
        self.evaluation=lift.evaluate_lift(self.rows,self.ids,.004)
        self.evaluation.update(task=lift.TASK,complete=True,label_eligible=True,total_physics_steps=self.n,dynamic_object_steps=self.n*3,gates=dict.fromkeys(audit.GATES,True),
            completed_phases=audit.PHASES,collection_candidate_sha256=self.candidate['candidate_sha256'],commands_sha256=self.command_binding['commands_sha256'],
            scene_sha256=audit.sha(self.scene),legacy_primary_task_success_used=False,state_writes_during_execution=0,bin_geometry_count=0,max_penetration_m=1e-7,max_all_object_illegal_support_force_n=0.,
            trace_sha256=audit.sha(self.execution/'MULTI_OBJECT_TRACE.json'),contacts_sha256=audit.sha(self.execution/'MULTI_OBJECT_CONTACTS.json'))
        self.persist_evaluation()

    def persist_evaluation(self):
        save(self.execution/'LIFT_EVALUATION.json',self.evaluation)
        label=int(self.evaluation['task_success'])
        save(self.trial/'COLLECTION_RECORD.json',dict(task=lift.TASK,candidate=self.candidate,command_binding=self.command_binding,
            evaluation_binding=bind(self.execution/'LIFT_EVALUATION.json'),formal_training_eligible=True,label=label))
        save(self.trial/'LIFT_PIPELINE_EXECUTION.json',dict(complete=True,new_labels=1,label=label))

    def run(self):return audit.audit_trial(self.trial,self.split)


class AuditTests(unittest.TestCase):
    def test_actual_other_body_success_and_valid_failure_zero(self):
        for success,compressed in ((True,True),(False,False)):
            with self.subTest(success=success),tempfile.TemporaryDirectory() as tmp:
                f=Fixture(tmp,success,compressed);result=f.run()
                self.assertEqual(result['status'],'pass',result.get('error'));self.assertIs(result['formal_training_eligible'],True)
                self.assertEqual(result['label'],int(success));self.assertEqual(result['runtime']['object_success']['primary'],False)
                self.assertEqual(result['runtime']['actual_command_coordinates'],28)
                self.assertEqual(result['original_object_ids'],f.ids)
                self.assertNotEqual(result['all_scene_source_mesh_sha256'],result['converted_scene_source_mesh_sha256'])

    def test_incomplete_infrastructure_missing_record_never_becomes_zero(self):
        for wrong in ('missing','incomplete','infra','gates'):
            with self.subTest(wrong=wrong),tempfile.TemporaryDirectory() as tmp:
                f=Fixture(tmp,False)
                if wrong=='missing':(f.trial/'COLLECTION_RECORD.json').unlink()
                else:
                    if wrong=='incomplete':f.evaluation['complete']=False
                    if wrong=='infra':f.evaluation['error']='infrastructure'
                    if wrong=='gates':f.evaluation['gates']['all_dynamic_objects_no_arm_support']=False
                    f.persist_evaluation()
                result=f.run();self.assertIsNone(result['label']);self.assertFalse(result['formal_training_eligible'])

    def test_full_source_vertex_bottom_and_raw_contact_force_are_independent(self):
        for wrong in ('bottom','finger','illegal'):
            with self.subTest(wrong=wrong),tempfile.TemporaryDirectory() as tmp:
                f=Fixture(tmp)
                if wrong=='bottom':f.rows[-1]['objects']['other']['source_bottom_m']+=.02
                else:
                    contact=f.contacts[-1]['object_contacts'][0]
                    if wrong=='finger':contact['normal_force_n']=.0001
                    else:contact['bodies'][1]='/World/Robot/armbody0'
                    save(f.execution/'MULTI_OBJECT_CONTACTS.json',dict(rows=f.contacts))
                    f.evaluation['contacts_sha256']=audit.sha(f.execution/'MULTI_OBJECT_CONTACTS.json')
                save(f.execution/'MULTI_OBJECT_TRACE.json',dict(rows=f.rows));f.evaluation['trace_sha256']=audit.sha(f.execution/'MULTI_OBJECT_TRACE.json');f.persist_evaluation()
                result=f.run();self.assertEqual(result['status'],'failed');self.assertIsNone(result['label'])

    def test_native_mimic_target_drift_rejected_even_when_saved_gates_are_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);f.native_arrays['joint_target'][-1,0]+=.01
            np.savez_compressed(f.execution/'trace.npz',**f.native_arrays)
            native=audit.read(f.execution/'evaluation.json');native['trace_sha256']=audit.sha(f.execution/'trace.npz');save(f.execution/'evaluation.json',native)
            result=f.run();self.assertEqual(result['status'],'failed');self.assertIn('command target',result['error'])

    def test_nonprimary_split_leak_and_converted_identity_mixup_rejected(self):
        for wrong in ('split','converted'):
            with self.subTest(wrong=wrong),tempfile.TemporaryDirectory() as tmp:
                f=Fixture(tmp);manifest=audit.read(f.split)
                if wrong=='split':manifest['internal_mesh_groups']['internal_check'].append(manifest['internal_mesh_groups']['fit'].pop(1))
                else:manifest['internal_mesh_groups']['fit'][1]['mesh_sha256']=f.objects[1]['source_mesh_sha256']
                save(f.split,manifest);result=f.run()
                self.assertEqual(result['status'],'failed');self.assertIsNone(result['label']);self.assertIn('Every original',result['error'])

    def test_lossless_pointer_content_tampering_rejected_and_old_json_stream_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp);pointer=audit.read(f.execution/'MULTI_OBJECT_CONTACTS.json')
            self.assertEqual(list(audit.iter_contact_rows(f.execution/'MULTI_OBJECT_CONTACTS.json')),f.contacts)
            with Path(pointer['data']['path']).open('ab') as output:output.write(b'changed')
            result=f.run();self.assertEqual(result['status'],'failed');self.assertIn('Contact data bytes',result['error'])

    def test_flat_pointer_source_bindings_and_contact_tick_count_are_enforced(self):
        for wrong in ('source','count'):
            with self.subTest(wrong=wrong),tempfile.TemporaryDirectory() as tmp:
                f=Fixture(tmp);path=f.execution/'MULTI_OBJECT_CONTACTS.json';pointer=audit.read(path)
                if wrong=='source':pointer['source_bindings']={str(f.robot):'0'*64}
                else:pointer['record_count']+=1
                save(path,pointer);f.evaluation['contacts_sha256']=audit.sha(path);f.persist_evaluation()
                result=f.run();self.assertEqual(result['status'],'failed');self.assertIsNone(result['label'])
                self.assertIn('SHA changed' if wrong=='source' else 'record count',result['error'])


if __name__=='__main__':unittest.main()
