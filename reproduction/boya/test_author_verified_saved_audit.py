"""Use the real audit ResourcePool, without Kit or numerical data replay."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import author_verified_saved_audit as V


class SavedAuditTests(unittest.TestCase):
    def test_real_pool_can_launch_audit_process_and_release_its_slot(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);directory=root/'audit';directory.mkdir()
            with V.R.lease(root/'DRIVER.lock')as fd:
                pool=V.R.ResourcePool(root/'slots',fd,{'path':'/fixture/protocol','sha256':'0'*64})
                with pool.stage(('audit',),directory):
                    process=pool.popen([sys.executable,'-I','-S','-B','-c','pass'],start_new_session=True)
                    self.assertEqual(process.wait(timeout=10),0)
            self.assertEqual(V.B.read(root/'slots/audit/0.json')['state'],'released')
            self.assertFalse(any((root/'slots/quarantine').glob('*.json')))
            self.assertEqual(V.R.LIMITS['audit'],2)

    def test_completed_existing_audit_cannot_be_repeated(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);execution=root/'execution';execution.mkdir()
            V.B.save(execution/'VERIFIED_FULL_RESULTS.json',dict(group=dict(status='full_complete'),audits=[dict(status='pass')]))
            with patch.object(V.F,'verify',return_value={}),self.assertRaises(ValueError):
                V.freeze(root/'audit',execution)
            self.assertFalse((root/'audit').exists())


if __name__=='__main__':unittest.main()
