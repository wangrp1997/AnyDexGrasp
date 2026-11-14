"""Batch scheduling/export contracts. Heavy stages are synthetic, without Kit."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

import author_collection_batch as B
import author_collection_export as E


def fixture(root, splits=('fit','fit','internal'),placement_null=(),capture_entry=None):
    mesh={}
    for split in ('fit','internal'):
        for i in range(3):
            p=root/f'{split}_{i}.mesh';p.write_text(split+str(i));mesh[split,i]=p
    rows=[]
    for index,split in enumerate(splits):
        scene_id=f'{split}_scene_{index}'
        objects=[dict(object_id=f'{split}_{i}',source_mesh=str(mesh[split,i]),
            source_mesh_sha256=B.sha(mesh[split,i]),source_visual_mesh=str(mesh[split,i]),
            source_visual_mesh_sha256=B.sha(mesh[split,i]))for i in range(3)]
        spec=root/f'{scene_id}.json'
        B.save(spec,dict(schema='anydex_boya_author_lift_scene_v1',task=B.TASK,scene_id=scene_id,
            status='placement_null'if index in placement_null else'ready',
            bin=None,objects=objects,target_object_id=objects[0]['object_id'],dataset_split=split))
        rows.append(dict(scene_id=scene_id,split=split,scene_spec=B.binding(spec),
            status='placement_null'if index in placement_null else'ready',
            source_object_ids=[o['object_id']for o in objects],
            source_object_digests={o['object_id']:o['source_visual_mesh_sha256']for o in objects}))
    splitfile=root/'split.json';B.save(splitfile,dict(fit=['fit_0','fit_1','fit_2'],internal=['internal_0','internal_1','internal_2']))
    queue=root/'QUEUE.json';B.save(queue,dict(schema=B.QUEUE_SCHEMA,seed=123,scenes=rows,
        split_manifest=B.binding(splitfile),inputs={'split':B.binding(splitfile)},sources={}))
    dummy=root/'dummy.json';B.save(dummy,{})
    output=root/'batch'
    protocol=B.freeze(queue,output,calibration=dummy,cloud_manifest=dummy,backend=dummy,
        checkpoint=dummy,scene_entry=Path(__file__),capture_entry=capture_entry or Path(__file__),
        execution_entry=Path(__file__),audit_entry=Path(__file__),fit_per_gesture=1,internal_per_gesture=1)
    return output,protocol


class FakeStages:
    def __init__(self,states=None,audit_status='pass'):
        self.calls=[];self.running=0;self.maximum=0;self.lock=threading.Lock()
        self.states=states or {g:'ready'for g in B.GESTURES}
        self.audit_status=audit_status

    def __call__(self,directory,command,expected,protocol):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        if (directory/'STAGE.json').exists():return B.read(directory/'STAGE.json')
        name=directory.name
        with self.lock:self.calls.append(name)
        result=dict(status='complete',complete=True,command=list(map(str,command)),outputs={})
        if name=='build':
            source=Path(command[command.index('--scene-spec')+1]);B.save(expected[0],B.read(source))
        elif name=='capture':
            source=Path(command[command.index('--scene')+1]);scene=B.read(source)
            B.save(expected[2],scene)
            B.save(expected[1],dict(scene_json=str(expected[2]),scene_sha256=B.sha(expected[2])))
            B.save(expected[0],dict(status='complete'))
            for p in expected[3:]:p.write_text('{}')
        elif name=='representation':
            B.save(expected[0],dict(network_forward_count=1,hand_head_load_count=0,completed_forward_objects=1))
            for p in expected[1:]:p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
        elif name.startswith('prepare_g'):
            g=int(name.removeprefix('prepare_g'))
            with self.lock:self.running+=1;self.maximum=max(self.maximum,self.running)
            time.sleep(.005)
            B.save(expected[0],dict(status=self.states[g]))
            with self.lock:self.running-=1
        elif name.startswith('execute_g'):
            B.save(expected[0],dict(status='physical_failure',physics_steps=23))
            B.save(expected[0].parent/'COLLECTION_RECORD.json',dict(label=0))
        elif name.startswith('audit_g'):
            scene_dir=directory.parents[1]
            row=next(r for r in protocol['queue']['scenes']if scene_dir.name.endswith(r['scene_id']))
            B.save(expected[0],dict(status=self.audit_status,formal_training_eligible=self.audit_status=='pass',
                label=0,split=row['split'],original_object_ids=row['source_object_ids'],
                all_scene_source_mesh_sha256=[row['source_object_digests'][i]for i in row['source_object_ids']]))
        else:raise AssertionError(name)
        result['outputs']={str(p):B.binding(p)for p in expected}
        B.save(directory/'STAGE.json',result)
        return result


def fake_trial(prepared,row):
    prepared=Path(prepared);g=int(prepared.name.removeprefix('g'));ids=sorted(row['source_object_ids'])
    paths={}
    for key in ('candidate','evaluation','commands','command_binding','scene'):
        p=prepared/(key+'.json');B.save(p,dict(kind=key));paths[key]=B.binding(p)
    return dict(record=B.binding(prepared/'COLLECTION_RECORD.json'),**paths,label=0,features=[.25]*480,target_index=7,
        scene_object_ids=ids,scene_mesh_sha256=[row['source_object_digests'][i]for i in ids],
        mesh_sha256=row['source_object_digests'][ids[0]],candidate_sha256='a'*64,gesture_id=g,actual_physics_steps=23)


class BatchTests(unittest.TestCase):
    def test_freeze_is_zero_dispatch_and_pins_venv_budget_entries_and_gpu0(self):
        with tempfile.TemporaryDirectory()as td,patch.object(B.subprocess,'Popen')as popen:
            output,p=fixture(Path(td));popen.assert_not_called()
            self.assertEqual(p['total_gesture_attempt_budget'],12);self.assertEqual(p['prepare_workers'],3)
            self.assertEqual(p['inputs']['python']['path'],str(Path(sys.executable).absolute()))
            self.assertEqual(p['env']['CUDA_VISIBLE_DEVICES'],'0');self.assertFalse(p['automatic_training'])
            self.assertFalse(p['old435_merged']);self.assertEqual(B.read(output/'PROGRESS.json')['ledger_rows'],0)

    def test_nonprimary_source_overlap_rejected_even_with_distinct_primary(self):
        with tempfile.TemporaryDirectory()as td:
            out,p=fixture(Path(td));q=copy.deepcopy(p['queue'])
            q['scenes'][-1]['source_object_digests']['internal_2']=q['scenes'][0]['source_object_digests']['fit_2']
            bad=Path(td)/'bad.json';B.save(bad,q)
            with self.assertRaises(ValueError):B.validate_queue(bad)

    def test_one_forward_per_scene_bounded_prepares_fixed_order_no_plan_not_negative(self):
        with tempfile.TemporaryDirectory()as td:
            out,p=fixture(Path(td));stages=FakeStages({1:'null_geometry',6:'ready',7:'infrastructure_failure',8:'no_candidate'})
            with patch.object(B,'_stage',side_effect=stages),patch.object(B,'validate_trial',side_effect=fake_trial):
                progress=B.run(out)
                calls=len(stages.calls);B.run(out)
            self.assertEqual(len(stages.calls),calls);self.assertEqual(stages.maximum,3)
            self.assertEqual(stages.calls.count('representation'),3)
            self.assertEqual(stages.calls.count('execute_g6'),2)
            rows=B.ledger(out,p)
            self.assertEqual(len(rows),12);self.assertEqual(sum(r['label']is not None for r in rows),2)
            self.assertTrue(all(r['label']is None for r in rows if r['gesture_id']!=6))
            self.assertEqual(progress['eligible_labels']['fit']['6'],1)
            self.assertEqual(progress['positive_labels']['fit']['6'],0)
            self.assertEqual(progress['status'],'budget_exhausted_insufficient_labels')

    def test_ready_executes_once_serial_and_quota_keeps_first_prefix_regardless_of_class(self):
        with tempfile.TemporaryDirectory()as td:
            out,p=fixture(Path(td));stages=FakeStages()
            with patch.object(B,'_stage',side_effect=stages),patch.object(B,'validate_trial',side_effect=fake_trial):
                result=B.run(out)
            self.assertTrue(result['quota_met']);self.assertEqual(result['actual_physics_steps'],8*23)
            self.assertEqual(result['captured_initialization_ticks'],1500)
            self.assertEqual(result['captured_initialization_object_steps'],4500)
            rows=B.ledger(out,p)
            self.assertTrue(all(r['status']=='quota_reached'for r in rows if r['queue_index']==1))
            self.assertEqual([r['gesture_id']for r in rows[:4]],list(B.GESTURES))
            self.assertEqual(stages.calls.count('capture'),2)

    def test_late_trial_verification_failure_keeps_physical_cost_without_label(self):
        with tempfile.TemporaryDirectory()as td:
            out,p=fixture(Path(td),('fit',));stages=FakeStages()
            with patch.object(B,'_stage',side_effect=stages),patch.object(B,'validate_trial',side_effect=ValueError('bad command SHA')):
                result=B.run(out)
            rows=B.ledger(out,p)
            self.assertEqual(result['actual_physics_steps'],4*23)
            self.assertTrue(all(r['status']=='infrastructure_failure'and r['label']is None for r in rows))

    def test_frozen_placement_null_never_builds_captures_or_creates_negative(self):
        with tempfile.TemporaryDirectory()as td:
            out,p=fixture(Path(td),('fit',),placement_null=(0,))
            with patch.object(B,'_stage')as stage:result=B.run(out)
            stage.assert_not_called()
            rows=B.ledger(out,p)
            self.assertEqual(len(rows),4)
            self.assertTrue(all(r['status']=='placement_null'and r['label']is None for r in rows))
            self.assertEqual(result['actual_physics_steps'],0)

    def test_runtime_audit_is_executed_and_bound_before_any_label_admission(self):
        with tempfile.TemporaryDirectory()as td:
            out,p=fixture(Path(td),('fit',));stages=FakeStages(audit_status='failed')
            with patch.object(B,'_stage',side_effect=stages),patch.object(B,'validate_trial',side_effect=fake_trial):
                result=B.run(out)
            rows=B.ledger(out,p)
            self.assertEqual(sum(name.startswith('audit_g')for name in stages.calls),4)
            self.assertTrue(all(r['status']=='infrastructure_failure'and r['label']is None for r in rows))
            self.assertTrue(all('runtime_audit'in r['artifact_bindings']for r in rows))
            self.assertEqual(result['actual_physics_steps'],92)

    def test_compact_capture_binds_full_initialization_stream_and_rechecks_on_resume(self):
        with tempfile.TemporaryDirectory()as td:
            out,p=fixture(Path(td),('fit',),capture_entry=B.HERE/'author_collection_capture.py');stages=FakeStages()
            with patch.object(B,'_stage',side_effect=stages),patch.object(B,'validate_trial',side_effect=fake_trial):B.run(out)
            row=B.ledger(out,p)[0];capture_stage=B.read(row['artifact_bindings']['capture_stage']['path'])
            names={Path(x).name for x in capture_stage['outputs']}
            self.assertTrue({'CONTACT_JOURNAL.json','initialization_contacts.jsonl.gz','initialization_trace.npz',
                'CONTACT_SENSOR_CHECK.json','render_geometry_check.json','native_rgbd.npz'}<=names)
            data=next(Path(x)for x in capture_stage['outputs']if Path(x).name=='initialization_contacts.jsonl.gz')
            data.write_text('corrupted')
            with self.assertRaises(ValueError):B.ledger(out,p)

    def test_interrupted_execution_has_unknown_cost_and_never_creates_negative(self):
        with tempfile.TemporaryDirectory()as td:
            out,p=fixture(Path(td),('fit',));stages=FakeStages()
            def interrupted(directory,command,expected,protocol):
                directory=Path(directory)
                if not directory.name.startswith('execute_g'):return stages(directory,command,expected,protocol)
                B.save(expected[0],dict(status='starting',physics_steps=0))
                result=dict(status='infrastructure_failure',command=command,outputs={},error='lost process')
                B.save(directory/'STAGE.json',result);return result
            with patch.object(B,'_stage',side_effect=interrupted):result=B.run(out)
            self.assertEqual(result['execution_starts_with_unknown_steps'],4)
            self.assertTrue(all(r['label']is None for r in B.ledger(out,p)))

    def test_completed_stage_is_reused_tampering_rejected_and_started_stage_never_retried(self):
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);output=root/'out.json';stage=root/'stage'
            protocol=dict(inputs={},sources={},stage_timeout_s=2.,env=B.DEFAULT_ENV,inherited_env={'PYTHONPATH':'','LD_LIBRARY_PATH':''})
            command=[sys.executable,'-c','from pathlib import Path; Path('+repr(str(output))+').write_text("done")']
            self.assertEqual(B._stage(stage,command,[output],protocol)['status'],'complete')
            with patch.object(B.subprocess,'Popen')as popen:B._stage(stage,command,[output],protocol);popen.assert_not_called()
            output.write_text('changed')
            with self.assertRaises(ValueError):B._stage(stage,command,[output],protocol)
            interrupted=root/'interrupted';interrupted.mkdir()
            B.save(interrupted/'STAGE.json',dict(command=command,status='running',child_pid=999999999))
            with patch.object(B.subprocess,'Popen')as popen:
                self.assertEqual(B._stage(interrupted,command,[output],protocol)['status'],'infrastructure_failure');popen.assert_not_called()
            self.assertFalse(B.read(interrupted/'STAGE.json')['resume_blocked'])

    def test_unknown_or_live_interrupted_child_blocks_further_stages(self):
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);stage=root/'stage';stage.mkdir();command=['unused']
            B.save(stage/'STAGE.json',dict(status='running',command=command))
            with patch.object(B.subprocess,'Popen')as popen,self.assertRaises(B.InterruptedChildUnresolved):
                B._stage(stage,command,[],{} )
            popen.assert_not_called();self.assertTrue(B.read(stage/'STAGE.json')['resume_blocked'])

    def test_source_mutation_in_subprocess_leaves_infrastructure_receipt(self):
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);source=root/'source';source.write_text('old');output=root/'out'
            protocol=dict(inputs={'source':B.binding(source)},sources={},stage_timeout_s=2.,env=B.DEFAULT_ENV,inherited_env={'PYTHONPATH':'','LD_LIBRARY_PATH':''})
            command=[sys.executable,'-c','from pathlib import Path; Path('+repr(str(source))+').write_text("changed"); Path('+repr(str(output))+').write_text("out")']
            result=B._stage(root/'stage',command,[output],protocol)
            self.assertEqual(result['status'],'infrastructure_failure');self.assertIn('dependency changed',result['error'])

    def test_export_new_only_nonpickle_full_scene_columns_and_independent_audit(self):
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);out,p=fixture(root);stages=FakeStages()
            with patch.object(B,'_stage',side_effect=stages),patch.object(B,'validate_trial',side_effect=fake_trial):B.run(out)
            rows=B.ledger(out,p);trials={str(Path(r['trial']['record']['path']).parent):r['trial']for r in rows if r['label']is not None}
            def verified(path,spec):return trials[str(path)]
            def audit(path,protocol):
                trial=trials[str(path)];row=next(r for r in rows if r.get('trial')==trial)
                return dict(status='pass',formal_training_eligible=True,label=0,split=row['split'],original_object_ids=trial['scene_object_ids'],all_scene_source_mesh_sha256=trial['scene_mesh_sha256'])
            with patch.object(B,'validate_trial',side_effect=verified),patch.object(E,'_audit_trial',side_effect=audit):
                manifest=E.export(out,root/'export')
            self.assertTrue(manifest['size_ready']);self.assertFalse(manifest['old435_merged'])
            for g in B.GESTURES:
                with np.load(root/f'export/gesture_{g}.npz',allow_pickle=False)as data:
                    self.assertEqual(data['features'].shape,(2,480));self.assertEqual(data['scene_mesh_sha256'].shape,(2,3))
                    self.assertEqual(data['scene_object_ids'].shape,(2,3));self.assertEqual(data['target_index'].tolist(),[7,7])
                    self.assertFalse(any(data[k].dtype.hasobject for k in data.files))

    def test_export_refuses_independent_audit_failure_without_training(self):
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);out,p=fixture(root,('fit',));stages=FakeStages()
            with patch.object(B,'_stage',side_effect=stages),patch.object(B,'validate_trial',side_effect=fake_trial):B.run(out)
            row=B.ledger(out,p)[0]
            with patch.object(B,'validate_trial',return_value=row['trial']),patch.object(E,'_audit_trial',return_value=dict(status='failed')):
                with self.assertRaises(ValueError):E.export(out,root/'export')
            self.assertFalse((root/'export/MANIFEST.json').exists())


if __name__=='__main__':unittest.main()
