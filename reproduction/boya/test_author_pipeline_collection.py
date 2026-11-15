"""Original stage contracts plus concurrent audit/ordered admission, no Kit."""
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import author_pipeline_collection as P
from test_author_collection_batch import fixture,fake_trial
from test_author_resident_collection import Stages


class ConcurrentStages(Stages):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.executing=0;self.execution_peak=0;self.auditing=0;self.audit_peak=0;self.overlap=False
        self.audit_calls=[]
    def __call__(self,directory,command,expected,protocol):
        if Path(directory).name=='resident_execute':
            targets=command[command.index('--prepared-dirs')+1:command.index('--output-dir')]
            if len(targets)!=1:raise AssertionError('Not one original trial per session')
            g=int(Path(targets[0]).name[1:])
            with self.lock:
                self.executing+=1;self.execution_peak=max(self.execution_peak,self.executing)
            try:
                time.sleep({1:.02,6:.08,7:.15,8:.24}[g])
                return super().__call__(directory,command,expected,protocol)
            finally:
                with self.lock:self.executing-=1
        if Path(directory).name.startswith('audit_g'):
            with self.lock:
                self.audit_calls.append(str(directory));self.auditing+=1
                self.audit_peak=max(self.audit_peak,self.auditing)
                self.overlap=self.overlap or self.executing>0
            try:
                time.sleep(.04)
                return super().__call__(directory,command,expected,protocol)
            finally:
                with self.lock:self.auditing-=1
        return super().__call__(directory,command,expected,protocol)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.addCleanup(self.temp.cleanup)
        self.smi=self.root/'nvidia-smi';self.smi.write_text('fixture identity only')
        self.identity=dict(index=0,uuid=P.GPU_UUID,pci_bus_id='fixture',name='fixture')

    def freeze(self,splits=('fit','fit'),placement_null=()):
        batch,_=fixture(self.root,splits=splits,placement_null=placement_null)
        root=self.root/'pipeline'
        with patch.object(P.shutil,'which',return_value=str(self.smi)),\
             patch.object(P.C,'gpu_identity',return_value=self.identity):
            protocol=P.freeze(batch,list(range(len(splits))),root)
        return root,protocol

    def run_fake(self,root,stages):
        original=P.build_namespace
        def namespace():
            ns=original();ns.update(LEGACY_LOCK=self.root/'gpu.lock',REGISTRY=self.root/'active.json')
            return ns
        with patch.object(P,'build_namespace',side_effect=namespace),\
             patch.object(P.C,'gpu_identity',return_value=self.identity),\
             patch.object(P.B,'_stage',stages),patch.object(P.B,'validate_trial',fake_trial),\
             patch.object(P.R,'install_worker_adapters',lambda *_:nullcontext()):
            return P.run(root)

    def test_freeze_has_eight_slots_two_audits_and_no_dispatch(self):
        with patch.object(P.B.subprocess,'Popen',side_effect=AssertionError('Freeze launched process')):
            root,p=self.freeze()
        self.assertEqual(p['resident_workers'],8);self.assertEqual(p['audit_workers'],2)
        self.assertEqual(p['total_execution_attempt_budget'],8)
        self.assertEqual(p['queue']['seed'],123)
        self.assertEqual(P.verify(root),p)
        self.assertEqual(P.R.LIMITS['kit'],3)
        self.assertFalse((root/'RUN_STARTED.json').exists())

    def test_actual_stage_handoff_overlaps_and_preserves_fixed_ledger(self):
        root,_=self.freeze();stages=ConcurrentStages();result=self.run_fake(root,stages)
        self.assertEqual(stages.execution_peak,8);self.assertEqual(stages.audit_peak,2)
        self.assertTrue(stages.overlap);self.assertEqual(len(set(stages.audit_calls)),8)
        self.assertEqual(len(stages.audit_calls),8)
        rows=P.build_namespace()['ledger'](root)
        self.assertEqual([(r['queue_index'],r['gesture_id']) for r in rows],
            [(i,g) for i in (0,1) for g in (1,6,7,8)])
        self.assertTrue(all(r['raw_label']==0 and r['label'] is None and r['exclude_training'] for r in rows))
        self.assertEqual(result['pipeline_validity'],'pass')
        self.assertEqual(result['known_final_execution_physics_steps'],184)
        self.assertEqual(result['captured_initialization_ticks'],1500)
        self.assertEqual(stages.model_loads,1);self.assertEqual(stages.forwards,2)
        pipeline=P.B.read(root/'AUDIT_PIPELINE_RESULTS.json')
        self.assertEqual(pipeline['queued_audits'],8);self.assertTrue(pipeline['all_futures_terminal'])

    def test_audit_failure_rejects_labels_and_preserves_execution_cost(self):
        root,_=self.freeze(splits=('fit',));stages=ConcurrentStages(audit_status='failed')
        result=self.run_fake(root,stages);rows=P.build_namespace()['ledger'](root)
        self.assertEqual(result['pipeline_validity'],'failed')
        self.assertTrue(all(r['raw_label'] is None for r in rows))
        self.assertEqual(result['known_final_execution_physics_steps'],92)
        self.assertEqual(len(stages.audit_calls),4)

    def test_nulls_are_not_negative_or_scheduled(self):
        root,_=self.freeze(placement_null=(1,))
        stages=ConcurrentStages(states={1:'ready',6:'null_geometry',7:'no_candidate',8:'infrastructure_failure'})
        result=self.run_fake(root,stages);rows=P.build_namespace()['ledger'](root)
        self.assertEqual(len(stages.groups),1);self.assertEqual(len(stages.audit_calls),1)
        self.assertEqual(sum(r['raw_label'] is not None for r in rows),1)
        self.assertEqual(sum(r['status']=='placement_null' for r in rows),4)

    def test_no_restart_after_started(self):
        root,_=self.freeze(splits=('fit',));stages=ConcurrentStages();self.run_fake(root,stages)
        before=len(stages.calls)
        with self.assertRaises(FileExistsError):self.run_fake(root,stages)
        self.assertEqual(len(stages.calls),before)

    def test_completed_job_survives_reported_session_close_failure(self):
        root,_=self.freeze(splits=('fit',));stages=ConcurrentStages(group_failure=True)
        result=self.run_fake(root,stages);rows=P.build_namespace()['ledger'](root)
        self.assertTrue(all(r['raw_label']==0 for r in rows))
        self.assertEqual(result['known_final_execution_physics_steps'],92)
        self.assertEqual(len(stages.audit_calls),4)

    def test_missing_finalization_keeps_unknown_cost_and_no_audit_label(self):
        root,_=self.freeze(splits=('fit',));stages=ConcurrentStages(unknown_group=True)
        result=self.run_fake(root,stages);rows=P.build_namespace()['ledger'](root)
        self.assertEqual(result['execution_starts_with_unknown_steps'],4)
        self.assertTrue(all(r['raw_label'] is None and r['execution_steps_unknown'] for r in rows))
        self.assertEqual(stages.audit_calls,[])

    def test_postexecution_source_change_cannot_publish_label_or_erase_cost(self):
        root,p=self.freeze(splits=('fit',));original=ConcurrentStages()
        modified=False
        def stage(directory,command,expected,protocol):
            nonlocal modified
            value=original(directory,command,expected,protocol)
            if Path(directory).name=='resident_execute' and not modified:
                modified=True
                Path(p['inputs']['calibration']['path']).write_text('changed after execution')
            return value
        with self.assertRaises(ValueError):self.run_fake(root,stage)
        self.assertFalse(any((root/'ledger').glob('*.json')))
        result=P.B.read(root/'RESIDENT_COLLECTION_RESULTS.json')
        self.assertEqual(result['known_final_execution_physics_steps'],92)
        self.assertEqual(len(result['unadmitted_execution_costs']),4)

    def test_cancel_records_intent_before_existing_owned_sigint_cleanup(self):
        process=type('Process',(),dict(pid=12345,wait=lambda self,**kw:(_ for _ in ()).throw(subprocess.TimeoutExpired('fixture',1)),
            poll=lambda self:None))()
        directory=self.root/'stage';directory.mkdir()
        def finish(*args):
            self.assertTrue((directory/'CANCELLATION.json').exists());return [{'signal':'SIGINT'}]
        with patch.object(P.R,'process_identity',return_value=dict(pid=12345,pgid=12345,start_ticks=1)),\
             patch.object(P.C,'finish_owned',side_effect=finish) as cleanup:
            wrapper=P.GracefulProcess(process,directory,{})
            with self.assertRaises(subprocess.TimeoutExpired):wrapper.wait(timeout=1)
        self.assertEqual(cleanup.call_count,1)

    def test_changed_pinned_original_is_rejected(self):
        with patch.object(P.B,'sha',return_value='changed'),self.assertRaisesRegex(ValueError,'source changed'):
            P.build_namespace()


if __name__=='__main__':unittest.main()
