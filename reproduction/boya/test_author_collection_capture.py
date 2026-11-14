"""Full contact preservation and original support/count semantics, without Kit."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import author_collection_capture as capture


class ContactJournalTests(unittest.TestCase):
    def test_round_trip_and_original_support_counts(self):
        roots=['/World/AuthorScene/a','/World/AuthorScene/b']
        frames=[]
        for step in range(1,751):
            contacts=[]
            for i,root in enumerate(roots):
                contacts.append(dict(bodies=[root,'/World/ground/collisionPlane'],
                    normal_force_n=.98 if (step+i)%7 else .0001, position_world_m=[.1,.2,.3]))
                contacts.append(dict(bodies=[root,roots[1-i]],normal_force_n=.2))
            frames.append(dict(step=step,contacts=contacts))
        with tempfile.TemporaryDirectory() as tmp:
            journal=capture.ContactJournal(tmp,roots)
            for row in frames:journal.append(row)
            journal.finish()
            with gzip.open(journal.path,'rt') as f:actual=[json.loads(line) for line in f]
            self.assertEqual(frames,actual)
            expected={root:dict(final_window_supported_ticks=sum(any(c['bodies'][0]==root
                and c['bodies'][1].startswith('/World/ground') and c['normal_force_n']>.001
                for c in row['contacts']) for row in frames[-250:]),
                contact_rows=sum(sum(c['bodies'][0]==root for c in row['contacts']) for row in frames)) for root in roots}
            self.assertEqual(journal.evidence(),expected)
            self.assertEqual(json.loads((Path(tmp)/'CONTACT_JOURNAL.json').read_text())['status'],'complete')

    def test_incomplete_and_nonconsecutive_not_silent_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal=capture.ContactJournal(tmp,['a'])
            with self.assertRaises(ValueError):journal.append(dict(step=2,contacts=[]))
            journal.append(dict(step=1,contacts=[]))
            with self.assertRaises(ValueError):journal.finish()
            journal.close()
            receipt=json.loads((Path(tmp)/'CONTACT_JOURNAL.json').read_text())
            self.assertEqual((receipt['frames'],receipt['status']),(1,'incomplete'))

    def test_overlay_matches_pinned_source_without_launching_kit(self):
        self.assertTrue(callable(capture.capture_function()))
        self.assertFalse(capture.overlay_receipt()['physics_or_admission_changes'])


if __name__=='__main__':unittest.main()
