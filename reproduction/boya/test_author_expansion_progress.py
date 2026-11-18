from pathlib import Path
import tempfile
import unittest
from author_expansion_progress import Progress


class ExpansionProgressTests(unittest.TestCase):
    def test_nulls_and_failed_returns_are_not_audited_samples(self):
        with tempfile.TemporaryDirectory() as d:
            p = Progress(Path(d),100)
            try:
                p.record(0,dict(status='placement_null',eligible=False,label=None))
                p.record(1,dict(status='grasp_failure',eligible=True,label=0))
                p.returned(1,False)
                with self.assertRaises(ValueError):
                    p.audited(0,[0])
                with self.assertRaises(ValueError):
                    p.audited(0,[1])
                self.assertEqual(p.summary()['null'],1)
                self.assertEqual(p.summary()['audited_labels'],0)
            finally:
                p.close()

    def test_real_labels_and_audit_counts_remain_distinct(self):
        with tempfile.TemporaryDirectory() as d:
            p = Progress(Path(d),100)
            try:
                p.record(0,dict(status='grasp_success',eligible=True,label=1))
                p.returned(0,True)
                self.assertEqual(p.summary()['audited_labels'],0)
                p.audited(0,[0])
                self.assertEqual(p.summary()['audited_labels'],1)
                with self.assertRaises(ValueError):
                    p.audited(0,[0])
                with self.assertRaises(ValueError):
                    p.record(0,dict(status='grasp_success',eligible=True,label=1))
                self.assertFalse(p.expired())
            finally:
                p.close()


if __name__=='__main__':
    unittest.main()
