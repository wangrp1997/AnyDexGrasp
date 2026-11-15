"""Real inherited protocol shape, without launching Kit or an audit worker."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import author_verified_full_control_v2 as V


class TmpfsControlTests(unittest.TestCase):
    def test_real_previous_protocol_gets_venv_interpreter_and_fresh_outputs(self):
        previous=V.B.read(V.B.HERE/'evidence/author_contact_integrity_20261115_v1/VERIFIED_FULL_PROTOCOL.json')
        self.assertNotIn('python',previous['inputs'])
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);ready=root/'ready.json';ready.write_text('{}')
            p=V.prepare_protocol(root,previous,ready)
            self.assertEqual(p['inputs']['python']['path'],str(Path(V.sys.executable).absolute()))
            V.B.verify({'actual_interpreter':p['inputs']['python']})
            self.assertEqual(p['env']['ANYDEX_CONTACT_SPOOL_ROOT'],V.SPOOL)
            self.assertEqual(p['total_worker_attempt_budget'],4)
            self.assertEqual(p['total_integrated_environment_steps'],12096)
            self.assertTrue(p['exclude_training']);self.assertEqual(p['retry_budget'],0)
            for i,w in enumerate(p['groups'][0]['workers']):
                self.assertEqual(w['output_dir'],str(root/'cpu_4'/f'worker_{i:02d}'/'result'))
                self.assertEqual(w['prepared_dir'],previous['prepared_template']['path'])
            self.assertNotIn('python',previous['inputs'])

    def test_failed_storage_readiness_cannot_create_a_physical_protocol(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);ready=root/'ready.json';V.B.save(ready,dict(status='failed'))
            with patch.object(V.OLD,'run',side_effect=AssertionError('Launched physics')):
                with self.assertRaises(ValueError):V.freeze(root/'control',ready)
            self.assertFalse((root/'control').exists())


if __name__=='__main__':unittest.main()
