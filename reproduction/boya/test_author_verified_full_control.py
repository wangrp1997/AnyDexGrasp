"""Bounded diagnostic budget and semantic audit admission; zero real physics."""
from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import author_verified_full_control as F

B=F.B


class FullControlTests(unittest.TestCase):
    def fixture(self,root,status='verified_storage_control'):
        reference=root/'old.json';dummy=root/'dummy';dummy.write_text('fixture')
        B.save(reference,dict(inputs={},sources={},prepared_template=dict(path=str(root/'template'),ready=B.binding(dummy),commands=B.binding(dummy)),
            gpu_uuid=F.STORAGE.P.GPU_UUID,gpu_identity={'uuid':F.STORAGE.P.GPU_UUID},
            env=dict(CUDA_VISIBLE_DEVICES=F.STORAGE.P.GPU_UUID,NVIDIA_VISIBLE_DEVICES='0'),inherited_env={}))
        stages=[]
        for mode in('standalone','kit'):
            path=root/f'{mode}.json'
            state=dict(timeline_playing=False,physics_scenes=[],simulation_context_exists=False)
            B.save(path,dict(mode=mode,groups=[dict(all_verified=True)],native_before=state,native_after=state))
            stages.append(dict(result=B.binding(path)))
        readiness=root/'READY.json';B.save(readiness,dict(schema=F.STORAGE.SCHEMA,status=status,physics_steps=0,actual_Kit_starts=1,stages=stages))
        return reference,readiness

    def test_freeze_exactly_four_old_trials_excluded_from_training(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);reference,readiness=self.fixture(root)
            with patch.object(F,'REFERENCE',reference),patch.object(F.STORAGE,'verify',return_value={}),\
                 patch.object(B.subprocess,'Popen',side_effect=AssertionError('Freeze launched native process')):
                p=F.freeze(root/'control',readiness)
            self.assertEqual(p['total_worker_attempt_budget'],4);self.assertEqual(p['total_integrated_environment_steps'],12096)
            self.assertTrue(p['exclude_training']);self.assertEqual(len(p['groups'][0]['workers']),4)
            self.assertFalse((root/'control/RUN_STARTED.json').exists());self.assertEqual(F.verify(root/'control'),p)

    def test_failed_storage_readiness_prevents_even_freezing_physical_protocol(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);reference,readiness=self.fixture(root,status='stopped')
            with patch.object(F,'REFERENCE',reference),patch.object(F.STORAGE,'verify',return_value={}),self.assertRaises(ValueError):
                F.freeze(root/'control',readiness)
            self.assertFalse((root/'control').exists())

    def test_exit_zero_but_failed_numeric_certificate_is_rejected(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp)
            def launch(command,**kwargs):
                B.save(Path(command[-1]),dict(status='failed'))
                return SimpleNamespace(pid=12345,wait=lambda **kw:0)
            pool=SimpleNamespace(stage=lambda *a:nullcontext(),popen=launch)
            p=dict(inputs={'python':{'path':'/fixture/python'}},sources={'author_verified_full_audit.py':{'path':'/fixture/audit.py'}},env={},inherited_env={})
            with patch.object(F.R,'process_identity',return_value=None):
                result=F.audit_one(root,dict(worker_id='worker_00',receipt={'path':'/fixture/result'}),p,pool)
            self.assertEqual(result['status'],'failed');self.assertEqual(result['returncode'],0)


if __name__=='__main__':unittest.main()
