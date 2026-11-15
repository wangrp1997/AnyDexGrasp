"""Only31 unstarted original draws, ragged final window, saved-audit export; no Kit."""
from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import author_production_continuation as D
from test_author_collection_batch import fixture,fake_trial
from test_author_production_collection import ProductionStages

B=D.B


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.addCleanup(self.temp.cleanup)
        self.smi=self.root/'nvidia-smi';self.smi.write_text('fixture')
        self.identity=dict(index=0,uuid=D.P.GPU_UUID,pci_bus_id='fixture',name='fixture')

    def freeze(self,nulls=()):
        batch,_=fixture(self.root,splits=('internal','fit')*15+('internal',),placement_null=nulls)
        base=B.read(batch/'BATCH_PROTOCOL.json');base['quotas']=dict(fit=1000,internal=200);B.save(batch/'BATCH_PROTOCOL.json',base)
        with patch.object(D.shutil,'which',return_value=str(self.smi)),patch.object(D.C,'gpu_identity',return_value=self.identity):
            p=D.freeze(batch,self.root/'remaining')
        return self.root/'remaining',p

    def run_fake(self,root,stages):
        with patch.object(B,'_stage',stages),patch.object(B,'validate_trial',fake_trial),\
             patch.object(D.R,'install_worker_adapters',lambda *_:nullcontext()),\
             patch.object(D.P.OLD,'LEGACY_LOCK',self.root/'gpu.lock'),patch.object(D.P.OLD,'REGISTRY',self.root/'registry.json'),\
             patch.object(D.C,'gpu_identity',return_value=self.identity):return D.run(root)

    def test_remaining_budget31_and_final_window7_without_dispatch(self):
        with patch.object(B.subprocess,'Popen',side_effect=AssertionError('Freeze dispatched')):root,p=self.freeze()
        self.assertEqual(p['total_scene_budget'],31);self.assertEqual(p['total_gesture_attempt_budget'],124)
        entries=B.read(p['window_index']['path'])['windows']
        self.assertEqual([len(e['source_scene_indices'])for e in entries],[8,8,8,7])
        self.assertEqual([r['split']for r in p['queue']['scenes']],['internal','fit']*15+['internal'])
        self.assertEqual(D.verify(root),p)

    def test_last_window_executes_and_original_indices_are_unique(self):
        root,p=self.freeze(nulls=tuple(range(30)));stages=ProductionStages();result=self.run_fake(root,stages)
        self.assertEqual(result['status'],'complete');rows=D.ledger(root,p)
        self.assertEqual(len(rows),124);self.assertEqual(sum(r['label']is not None for r in rows),4)
        self.assertEqual({r['queue_index']for r in rows},set(range(31)))
        self.assertEqual(len(stages.groups),4);self.assertEqual(result['known_final_execution_physics_steps'],92)
        with patch.object(B,'validate_trial',fake_trial):export=D.export(root,self.root/'export')
        self.assertEqual([d['counts']for d in export['datasets'].values()],[dict(fit=0,internal=1)]*4)

    def test_failure_still_stops_later_windows_without_retry(self):
        root,p=self.freeze();stages=ProductionStages(fail_capture=True);result=self.run_fake(root,stages)
        self.assertEqual(result['status'],'stopped');self.assertEqual(stages.calls.count('capture'),1)
        self.assertFalse((root/'windows/01/RUN_STARTED.json').exists())

    def test_original_production_protocol_still_has32_scene_contract(self):
        root,p=self.freeze()
        with self.assertRaises(ValueError):D.ORIGINAL.verify(root)
        self.assertEqual(D.ORIGINAL.SCHEMA,'anydex_boya_author_production_pilot_v1')


if __name__=='__main__':unittest.main()
