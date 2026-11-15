"""Original fifteen-scene tail and verified admission; no real physics."""
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import author_production_tail as D
from test_author_collection_batch import fixture,fake_trial
from test_author_production_collection import ProductionStages

B=D.B


class TailTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.addCleanup(self.temp.cleanup)
        self.smi=self.root/'smi';self.smi.write_text('fixture')
        self.identity=dict(uuid=D.P.GPU_UUID)

    def freeze(self,nulls=()):
        batch,_=fixture(self.root,splits=('internal','fit')*7+('internal',),placement_null=nulls)
        base=B.read(batch/'BATCH_PROTOCOL.json');base['quotas']=dict(fit=1000,internal=200);B.save(batch/'BATCH_PROTOCOL.json',base)
        with patch.object(D.shutil,'which',return_value=str(self.smi)),patch.object(D.C,'gpu_identity',return_value=self.identity):
            p=D.freeze(batch,self.root/'tail')
        return self.root/'tail',p

    def execute(self,root,stages):
        with patch.object(B,'_stage',stages),patch.object(B,'validate_trial',fake_trial),\
             patch.object(D.R,'install_worker_adapters',lambda *_:nullcontext()),\
             patch.object(D.P.OLD,'LEGACY_LOCK',self.root/'gpu.lock'),patch.object(D.P.OLD,'REGISTRY',self.root/'registry.json'),\
             patch.object(D.C,'gpu_identity',return_value=self.identity),\
             patch.object(D,'verified_contact_artifacts',return_value={}):return D.run(root)

    def test_freeze_exact15_with_verified_entry_and_windows8_7(self):
        with patch.object(B.subprocess,'Popen',side_effect=AssertionError('Freeze dispatched')):root,p=self.freeze()
        self.assertEqual(p['total_scene_budget'],15);self.assertEqual(p['total_gesture_attempt_budget'],60)
        entries=B.read(p['window_index']['path'])['windows']
        self.assertEqual([len(x['source_scene_indices'])for x in entries],[8,7])
        for entry in entries:
            cp=D.verify_window(Path(entry['protocol']['path']).parent)
            self.assertEqual(cp['sources']['author_resident_execution.py'],B.binding(D.EXECUTION.__file__))
            self.assertEqual(cp['env']['ANYDEX_CONTACT_SPOOL_ROOT'],D.SPOOL)
        self.assertEqual(D.verify(root),p);self.assertFalse((root/'RUN_STARTED.json').exists())

    def test_ragged_last_window_and_original_export_no_second_audit(self):
        root,p=self.freeze(tuple(range(14)));stages=ProductionStages();result=self.execute(root,stages)
        self.assertEqual(result['status'],'complete');self.assertEqual(result['terminal_slots'],60)
        self.assertEqual(len(stages.groups),4);self.assertEqual(result['known_final_execution_physics_steps'],92)
        with patch.object(B,'validate_trial',fake_trial),patch.object(B.subprocess,'Popen',side_effect=AssertionError('Export dispatched')):
            exported=D.export(root,self.root/'export')
        self.assertEqual([x['counts']for x in exported['datasets'].values()],[dict(fit=0,internal=1)]*4)

    def test_exit0_failed_certificate_stops_later_window_without_retry(self):
        root,p=self.freeze(tuple(range(1,15)));stages=ProductionStages(audit_status='failed')
        result=self.execute(root,stages)
        self.assertEqual(result['status'],'stopped');self.assertTrue((root/'STOP_INTENT.json').exists())
        self.assertFalse((root/'windows/01/RESIDENT_COLLECTION_RESULTS.json').exists())
        self.assertTrue(all(r['label']is None for r in D.ledger(root,p)))

    def test_seal_and_original_counter_guard_use_actual_journal(self):
        prepared=self.root/'prepared';directory=prepared/'execution';directory.mkdir(parents=True)
        journal=D.JOURNAL.ContactJournal(directory/'MULTI_OBJECT_CONTACTS.jsonl.gz',spool_root=self.root/'raw')
        journal.append(dict(step=1,contacts=[dict(force=0.)]));B.save(directory/'MULTI_OBJECT_CONTACTS.json',journal.pointer())
        B.save(directory/'CONTACT_STORAGE_COST.json',dict(status='original_finish_returned',original_runtime_steps=1,
            observed_trace_rows=1,contact_records=1))
        trial=dict(record=dict(path=str(prepared/'COLLECTION_RECORD.json')),actual_physics_steps=1)
        self.assertIn('contact_integrity_seal',D.verified_contact_artifacts(trial))
        with (directory/'MULTI_OBJECT_CONTACTS.jsonl.gz').open('ab')as f:f.write(b'changed')
        with self.assertRaises(D.JOURNAL.JournalIntegrityError):D.verified_contact_artifacts(trial)

    def test_real_expanded_pool_supports_audit_slots(self):
        protocol=self.root/'pool.json';B.save(protocol,dict(stage_timeout_s=10))
        directory=self.root/'audit';directory.mkdir()
        with D.R.lease(self.root/'driver.lock')as fd:
            pool=D.P.ExpandedPool(self.root/'slots',fd,B.binding(protocol))
            with pool.stage(('audit',),directory):
                with(directory/'process.log').open('xb')as log:
                    process=pool.popen([sys.executable,'-I','-S','-B','-c','pass'],stdout=log,start_new_session=True)
                    self.assertEqual(process.wait(timeout=10),0)
        self.assertEqual(B.read(self.root/'slots/audit/0.json')['state'],'released')


if __name__=='__main__':unittest.main()
