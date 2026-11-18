from pathlib import Path
import tempfile
import unittest
from author_hour_metrics import HourMetrics


class HourWindowTests(unittest.TestCase):
    def setup_metrics(self,d):
        self.now=100
        return HourMetrics(Path(d),dict(started_monotonic_ns=100,deadline_monotonic_ns=1100),clock_ns=lambda:self.now)

    def test_record_return_audit_are_separate_and_deadline_is_exclusive(self):
        with tempfile.TemporaryDirectory()as d:
            m=self.setup_metrics(d);self.now=1000
            m.record(0,dict(status='grasp_success',eligible=True,label=1));m.returned(0,True)
            self.now=1099;m.audited(0,[0]);self.now=1100
            m.record(1,dict(status='grasp_failure',eligible=True,label=0));m.returned(1,True);m.audited(1,[1])
            s=m.summary();self.assertEqual(s['eligible_labels_within_window'],1)
            self.assertEqual(s['audited_labels_within_window'],1);self.assertEqual(s['timed_out_tail_labels'],1)
            self.assertTrue(s['window_elapsed']);m.close()

    def test_late_audit_does_not_retroactively_count_as_verified(self):
        with tempfile.TemporaryDirectory()as d:
            m=self.setup_metrics(d);m.record(0,dict(status='grasp_failure',eligible=True,label=0));m.returned(0,True)
            self.now=1101;m.audited(0,[0]);s=m.summary()
            self.assertEqual(s['grasp_and_return_within_window'],1);self.assertEqual(s['audited_labels_within_window'],0)
            self.assertEqual(s['audited_labels_including_drain'],1);m.close()

    def test_nulls_and_duplicate_records_cannot_inflate_counts(self):
        with tempfile.TemporaryDirectory()as d:
            m=self.setup_metrics(d);m.record(0,dict(status='no_candidate',eligible=False,label=None))
            with self.assertRaises(ValueError):m.record(0,dict(status='grasp_success',eligible=True,label=1))
            with self.assertRaises(ValueError):m.audited(0,[0])
            self.assertEqual(m.summary()['eligible_labels_within_window'],0);m.close()

    def test_failed_return_cannot_be_counted_as_audited_complete_sample(self):
        with tempfile.TemporaryDirectory()as d:
            m=self.setup_metrics(d);m.record(0,dict(status='grasp_success',eligible=True,label=1));m.returned(0,False)
            with self.assertRaises(ValueError):m.audited(0,[0])
            self.assertEqual(m.summary()['grasp_and_return_within_window'],0);m.close()


if __name__=='__main__':unittest.main()
