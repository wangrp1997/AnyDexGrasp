"""Production admission, bounded dispatch and immutable diagnostic separation; no Kit."""
from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import author_production_collection as D
from test_author_collection_batch import fixture,fake_trial
from test_author_resident_collection import Stages

B=D.B


class ProductionStages(Stages):
    def __init__(self,fail_capture=False,**kwargs):
        super().__init__(**kwargs);self.fail_capture=fail_capture
    def __call__(self,directory,command,expected,protocol):
        directory=Path(directory)
        if directory.name=='capture_audit':
            directory.mkdir(parents=True,exist_ok=True)
            B.save(expected[0],dict(status='pass',physics_ticks=750,all_contact_frames=750,
                inputs_sha256={}))
            result=dict(status='complete',outputs={str(expected[0]):B.binding(expected[0])})
            B.save(directory/'STAGE.json',result);self.calls.append('capture_audit');return result
        result=super().__call__(directory,command,expected,protocol)
        if directory.name=='capture':
            B.save(Path(expected[0]),dict(status='complete',physics_ticks=750))
            result['outputs'][str(expected[0])]=B.binding(expected[0]);B.save(directory/'STAGE.json',result)
            if self.fail_capture:
                result['status']='infrastructure_failure';B.save(directory/'STAGE.json',result)
        return result


class ProductionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.addCleanup(self.temp.cleanup)
        self.smi=self.root/'nvidia-smi';self.smi.write_text('fixture GPU0 identity')
        self.identity=dict(index=0,uuid=D.P.GPU_UUID,pci_bus_id='fixture',name='fixture')

    def test_durable_stop_request_blocks_new_work_but_allows_audit_drain(self):
        control=D.StopControl(self.root);B.save(self.root/'STOP_REQUEST.json',dict(reason='user stop'))
        with self.assertRaises(D.ProductionStopped):control.check()
        control.check(audit=True)
        self.assertTrue((self.root/'STOP_INTENT.json').exists())

    def frozen(self,placement_null=()):
        batch,_=fixture(self.root,splits=('fit','internal')*16,placement_null=placement_null)
        # Keep the real eventual target, never use tiny pilot counts as readiness.
        p=B.read(batch/'BATCH_PROTOCOL.json');p['quotas']=dict(fit=1000,internal=200);B.save(batch/'BATCH_PROTOCOL.json',p)
        with patch.object(D.shutil,'which',return_value=str(self.smi)),patch.object(D.C,'gpu_identity',return_value=self.identity):
            protocol=D.freeze(batch,self.root/'production')
        return self.root/'production',protocol

    def execute(self,root,stages):
        with patch.object(B,'_stage',stages),patch.object(B,'validate_trial',fake_trial),\
             patch.object(D.R,'install_worker_adapters',lambda *_:nullcontext()),\
             patch.object(D.P.OLD,'LEGACY_LOCK',self.root/'gpu.lock'),patch.object(D.P.OLD,'REGISTRY',self.root/'registry.json'),\
             patch.object(D.C,'gpu_identity',return_value=self.identity):
            return D.run(root)

    def test_freeze32_interleaved128_slots_four_windows_no_dispatch(self):
        with patch.object(B.subprocess,'Popen',side_effect=AssertionError('Freeze launched process')):
            root,p=self.frozen()
        self.assertEqual(p['total_gesture_attempt_budget'],128);self.assertFalse(p['exclude_training'])
        self.assertEqual([r['split']for r in p['queue']['scenes']],['fit','internal']*16)
        self.assertFalse((root/'RUN_STARTED.json').exists())
        self.assertEqual(D.verify(root),p)
        self.assertEqual(D.R.LIMITS['kit'],3)

    def test_new_production_labels_and_once_only_original_certificates(self):
        root,p=self.frozen(placement_null=tuple(range(2,32)));stages=ProductionStages()
        result=self.execute(root,stages);rows=D.ledger(root,p)
        self.assertEqual(result['status'],'complete');self.assertEqual(len(rows),128)
        self.assertEqual(sum(r['label']is not None for r in rows),8)
        self.assertTrue(all(r['formal_training_eligible']and not r['exclude_training']for r in rows if r['label']is not None))
        self.assertEqual(sum(r['status']=='placement_null'for r in rows),120)
        self.assertEqual(stages.calls.count('capture_audit'),2)
        self.assertEqual(sum(n.startswith('audit_g')for n in stages.calls),8)
        self.assertEqual(len(stages.groups),8)
        self.assertEqual(result['known_final_execution_physics_steps'],184)
        self.assertEqual(result['captured_initialization_ticks'],1500)

    def test_null_geometry_and_no_candidate_are_never_negative_labels(self):
        root,p=self.frozen(placement_null=tuple(range(2,32)))
        result=self.execute(root,ProductionStages(states={1:'ready',6:'null_geometry',7:'no_candidate',8:'null_geometry'}))
        rows=D.ledger(root,p);self.assertEqual(sum(r['label']is not None for r in rows),2)
        self.assertEqual(result['execution_starts_with_unknown_steps'],0)

    def test_first_capture_failure_stops_new_dispatch_and_later_windows(self):
        root,_=self.frozen();stages=ProductionStages(fail_capture=True);result=self.execute(root,stages)
        self.assertEqual(result['status'],'stopped');self.assertTrue((root/'STOP_INTENT.json').exists())
        self.assertEqual(stages.calls.count('capture'),1);self.assertEqual(stages.groups,[])
        self.assertFalse((root/'windows/01/RUN_STARTED.json').exists())
        self.assertEqual(result['windows_finished'],1)

    def test_failed_first_audit_never_admits_a_label_and_keeps_execution_cost(self):
        root,p=self.frozen(placement_null=tuple(range(1,32)));stages=ProductionStages(audit_status='failed')
        result=self.execute(root,stages);rows=D.ledger(root,p)
        self.assertEqual(result['status'],'stopped');self.assertTrue(all(r['label']is None for r in rows))
        self.assertEqual(result['known_final_execution_physics_steps'],92)
        self.assertFalse((root/'windows/01/RUN_STARTED.json').exists())

    def test_no_restart_or_budget_extension_after_run_started(self):
        root,_=self.frozen(placement_null=tuple(range(32)));stages=ProductionStages();self.execute(root,stages)
        before=list(stages.calls)
        with self.assertRaises(FileExistsError):self.execute(root,stages)
        self.assertEqual(stages.calls,before)

    def test_changed_input_fails_before_any_stage_or_run_marker(self):
        root,p=self.frozen();Path(p['inputs']['backend']['path']).write_text('changed')
        stages=ProductionStages()
        with self.assertRaises(ValueError):self.execute(root,stages)
        self.assertFalse((root/'RUN_STARTED.json').exists());self.assertEqual(stages.calls,[])

    def test_export_saved_certificates_without_second_point_decode_or_training(self):
        root,p=self.frozen(placement_null=tuple(range(2,32)));stages=ProductionStages();self.execute(root,stages)
        with patch.object(B,'validate_trial',fake_trial),\
             patch.object(B.subprocess,'Popen',side_effect=AssertionError('Export started a process')):
            manifest=D.export(root,self.root/'export')
        self.assertFalse(manifest['size_ready']);self.assertFalse(manifest['old435_merged'])
        self.assertEqual(manifest['training_updates'],0)
        self.assertEqual(len(manifest['audits']),8)
        for entry in manifest['datasets'].values():
            self.assertEqual(entry['counts'],dict(fit=1,internal=1))
            with np.load(entry['binding']['path'],allow_pickle=False)as data:
                self.assertEqual(data['scene_object_ids'].shape,(2,3));self.assertEqual(data['features'].shape,(2,480))
                self.assertEqual(data['labels'].tolist(),[0,0])

    def test_parent_tampered_label_cannot_be_exported(self):
        root,p=self.frozen(placement_null=tuple(range(1,32)));self.execute(root,ProductionStages())
        path=root/'ledger/0000000_g1.json';row=B.read(path);row['label']=1;B.save(path,row)
        with self.assertRaises(ValueError):D.ledger(root,p)

    def test_parent_cannot_override_a_child_training_exclusion(self):
        root,p=self.frozen(placement_null=tuple(range(1,32)));self.execute(root,ProductionStages())
        path=root/'ledger/0000000_g1.json';row=B.read(path);rawpath=Path(row['raw_ledger']['path'])
        raw=B.read(rawpath);raw['exclude_training']=True;B.save(rawpath,raw)
        row['exclude_training']=True;row['raw_ledger']=B.binding(rawpath);B.save(path,row)
        with self.assertRaisesRegex(ValueError,'diagnostic or uncertified'):D.ledger(root,p)

    def test_postexecution_integrity_failure_retains_unadmitted_cost(self):
        root,p=self.frozen(placement_null=tuple(range(1,32)))
        class Changed(ProductionStages):
            def __call__(self,directory,command,expected,protocol):
                result=super().__call__(directory,command,expected,protocol)
                if Path(directory).name=='resident_execute':Path(p['inputs']['backend']['path']).write_text('changed after execution')
                return result
        result=self.execute(root,Changed())
        self.assertEqual(result['status'],'stopped')
        self.assertGreater(result['known_final_execution_physics_steps'],0)
        self.assertFalse(any((root/'ledger').glob('*.json')))
        self.assertFalse((root/'windows/01/RUN_STARTED.json').exists())

    def test_original_diagnostic_namespace_still_rejects_production_protocol(self):
        root,p=self.frozen();directory=root/'windows/00'
        with self.assertRaises(FileNotFoundError):D.P.verify(directory)
        self.assertEqual(D.P.OLD.SCHEMA,'anydex_boya_author_resident_collection_v1')


if __name__=='__main__':unittest.main()
