"""An exit0/status-failed numeric certificate must stop new physical dispatch."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import author_production_control_v2 as D
from test_author_collection_batch import fixture

B=D.B


class CertificateStopTests(unittest.TestCase):
    def test_v2_freeze_binds_original_source_and_wires_semantic_guard_without_dispatch(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);batch,_=fixture(root,splits=('fit','internal')*16)
            base=B.read(batch/'BATCH_PROTOCOL.json');base['quotas']=dict(fit=1000,internal=200);B.save(batch/'BATCH_PROTOCOL.json',base)
            smi=root/'smi';smi.write_text('fixture')
            with patch.object(D.C,'gpu_identity',return_value={'uuid':D.P.GPU_UUID}),\
                 patch.object(D.shutil,'which',return_value=str(smi)),\
                 patch.object(B.subprocess,'Popen',side_effect=AssertionError('Freeze dispatched')):
                p=D.freeze(batch,root/'v2')
            self.assertEqual(p['schema'],'anydex_boya_author_production_pilot_v2')
            self.assertEqual(D.verify(root/'v2'),p)
            self.assertFalse((root/'v2/RUN_STARTED.json').exists())
            self.assertIs(D.run.__globals__['guarded_stages'],D.semantic_guarded_stages)
            self.assertIn('frozen_production_v1',p['sources'])

    def test_certificate_semantics_stop_dispatch_and_allow_audit_drain(self):
        for status,label,eligible,stops in [('pass',1,True,False),('pass',0,True,False),
            ('failed',None,False,True),('pass',None,True,True),('pass',True,True,True)]:
            with self.subTest(status=status,label=label),tempfile.TemporaryDirectory()as temp:
                root=Path(temp);control=D.StopControl(root);output=root/'AUDIT.json'
                calls=[]
                def direct(directory,command,expected,protocol):
                    calls.append(list(command));B.save(output,dict(status=status,label=label,formal_training_eligible=eligible))
                    return dict(status='complete',returncode=0)
                with patch.object(B,'_stage',direct),D.semantic_guarded_stages(control):
                    B._stage(root/'audit',['python','audit_author_collection_runtime.py'],[output],{})
                    self.assertEqual(control.event.is_set(),stops)
                    if stops:
                        with self.assertRaises(D.ProductionStopped):
                            B._stage(root/'capture',['python','author_collection_capture.py'],[output],{})
                        self.assertEqual(len(calls),1)
                        B._stage(root/'audit2',['python','audit_author_collection_runtime.py'],[output],{})
                        self.assertEqual(len(calls),2)


if __name__=='__main__':unittest.main()
