"""Storage protocol guards and exact full-row journal API; no simulator imports."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import author_storage_path_validation as V


class StoragePathTests(unittest.TestCase):
    def test_non_tmpfs_memory_path_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory()as temp:
            source=Path(temp)/'source';source.write_bytes(b'full fixture\n')
            expected=dict(sha256=V.W.sha(source),bytes=source.stat().st_size,records=1)
            with patch.object(V,'SOURCE',source),patch.object(V,'EXPECTED',expected),\
                 patch.object(V,'filesystem',return_value=dict(fstype='zfs')):
                with self.assertRaises(ValueError):V.freeze(Path(temp)/'control')
            self.assertFalse((Path(temp)/'control').exists())

    def test_changed_full_source_is_rejected_before_any_worker(self):
        with tempfile.TemporaryDirectory()as temp:
            source=Path(temp)/'source';source.write_bytes(b'changed')
            with patch.object(V,'SOURCE',source),self.assertRaises(ValueError):
                V.freeze(Path(temp)/'control')

    def test_worker_uses_complete_journal_and_two_actual_isolated_readbacks(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);source=root/'source'
            rows=[dict(step=i,points=[dict(force=0.,extra=[1.,-0.,1e-20])],objects=[{},{}])for i in range(1,4)]
            source.write_bytes(b''.join((json.dumps(r,separators=(',',':'),allow_nan=False)+'\n').encode()for r in rows))
            p=dict(source=V.binding(source),expected=dict(sha256=V.W.sha(source),bytes=source.stat().st_size,records=3),
                   sources={},paths=dict(disk=str(root/'spool')),python=V.binding(V.sys.executable))
            with patch.object(V,'verify',return_value=p):result=V.worker(root,'disk',0)
            self.assertEqual(result['status'],'verified');self.assertEqual(result['logical'],p['expected'])
            directory=root/'disk/writer_00'
            V.independent_readback(directory/'CONTACTS.jsonl.gz',directory/'FINAL_READBACK.json',p)
            value=V.read(directory/'FINAL_READBACK.json')
            self.assertTrue(value['isolated']);self.assertTrue(value['no_site'])
            self.assertEqual(value['logical_readback'],p['expected']);self.assertEqual(value['physics_steps'],0)


if __name__=='__main__':unittest.main()
