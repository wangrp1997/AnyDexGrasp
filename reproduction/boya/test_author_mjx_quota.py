import hashlib
from pathlib import Path
import tempfile
import unittest

from author_scene_prepare_cache import PreparationCache
from author_mjx_quota_progress import QuotaProgress


class QuotaTests(unittest.TestCase):
    def metrics(self,path):
        return QuotaProgress(path,dict(rows=[dict(index=0,split='fit'),dict(index=1,split='internal'),dict(index=2,split='fit')],
            existing_fit_records=99,existing_internal_records=4,target_fit_records=100))

    def test_negative_fit_counts_only_after_return_and_audit(self):
        with tempfile.TemporaryDirectory()as d:
            m=self.metrics(Path(d))
            try:
                m.record(0,dict(status='grasp_failure',label=0,eligible=True))
                self.assertFalse(m.runtime_target_ready());m.returned(0,True)
                self.assertTrue(m.runtime_target_ready());self.assertFalse(m.summary()['fit_target_audited'])
                m.audited(0,[0]);self.assertTrue(m.summary()['fit_target_audited'])
            finally:m.close()

    def test_internal_and_unstarted_slots_do_not_inflate_fit_or_processed_nulls(self):
        with tempfile.TemporaryDirectory()as d:
            m=self.metrics(Path(d))
            try:
                m.record(1,dict(status='grasp_success',label=1,eligible=True));m.returned(1,True);m.audited(0,[1])
                m.record(2,dict(status='quota_reached_not_started',label=None,eligible=False))
                self.assertFalse(m.runtime_target_ready());s=m.summary()
                self.assertEqual(s['processed_terminal_scenes'],1);self.assertEqual(s['processed_null_scenes'],0)
                self.assertEqual(s['unstarted_slots'],1);self.assertEqual(s['cumulative_audited_internal_records'],5)
            finally:m.close()

    def test_preparation_cache_rechecks_changed_file_and_rejects_wrong_hash(self):
        with tempfile.TemporaryDirectory()as d:
            path=Path(d)/'input';path.write_bytes(b'first');expected=hashlib.sha256(b'first').hexdigest()
            c=PreparationCache();c.verify({str(path):expected});c.verify({str(path):expected});self.assertEqual(c.hashes,2)
            path.write_bytes(b'other')
            with self.assertRaises(ValueError):c.verify({str(path):expected})
            self.assertEqual(c.hashes,3)


if __name__=='__main__':unittest.main()
