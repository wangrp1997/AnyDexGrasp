"""Native import probe inherits the verified environment before physical dispatch."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import production_dependency_environment as E

B=E.B


class EnvironmentTests(unittest.TestCase):
    def test_reference_environment_restored_and_failed_import_saved(self):
        for code in(0,1):
            with self.subTest(returncode=code),tempfile.TemporaryDirectory()as temp:
                root=Path(temp);batch=root/'batch';batch.mkdir()
                B.save(batch/'BATCH_PROTOCOL.json',dict(inputs={'python':dict(path='/fixture/python',sha256='fixture')},sources={},
                    env=dict(CUDA_VISIBLE_DEVICES='0'),inherited_env=dict(PYTHONPATH='wrong',LD_LIBRARY_PATH='')))
                ref=root/'reference.json';B.save(ref,dict(inputs={},sources={},inherited_env=dict(PYTHONPATH='verified-overlay',LD_LIBRARY_PATH='verified-libraries')))
                captured={}
                def probe(command,**kwargs):
                    captured.update(kwargs['env']);return SimpleNamespace(returncode=code,stdout='fixture',stderr='missing'if code else'')
                with patch.object(E.subprocess,'run',side_effect=probe):
                    if code:
                        with self.assertRaises(RuntimeError):E.restore_and_probe(batch,ref,root/'PREFLIGHT.json')
                    else:E.restore_and_probe(batch,ref,root/'PREFLIGHT.json')
                self.assertTrue(captured['PYTHONPATH'].endswith('verified-overlay'))
                self.assertEqual(captured['LD_LIBRARY_PATH'],'verified-libraries')
                self.assertEqual(B.read(root/'PREFLIGHT.json')['returncode'],code)
                self.assertFalse((batch/'RUN_STARTED.json').exists())


if __name__=='__main__':unittest.main()
