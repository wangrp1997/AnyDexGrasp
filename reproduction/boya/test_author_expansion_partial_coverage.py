import json
from pathlib import Path
import tempfile
import unittest

import author_collection_batch as B
from finish_author_expansion_metadata import coverage


class PartialCoverageTests(unittest.TestCase):
    def test_incomplete_execution_without_outcomes_stays_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);catalog=root/'catalog.json'
            B.save(catalog,dict(objects={k:{} for k in ('a','b','c')},
                split_object_ids=dict(fit=['a','b','c'],internal=[])))
            B.save(root/'PROTOCOL.json',dict(inputs=dict(expansion_catalog=B.binding(catalog)),
                rows=[dict(index=0,source_object_ids=['a','b','c'],scene={'path':'source'})]))
            directory=root/'scenes/00';trace=directory/'execution/TRACE.jsonl'
            trace.parent.mkdir(parents=True)
            trace.write_text(json.dumps(dict(objects={k:dict(finger_loads_n=[0]) for k in ('a','b','c')},contacts=[]))+'\n')
            B.save(directory/'RECORD.json',dict(eligible=False,label=None))
            B.save(directory/'execution/RESULTS.json',dict(trace=B.binding(trace),
                status='physical_stop',evaluation=dict(task_success=False,incomplete=True)))
            result=coverage(root)['objects']
            for row in result.values():
                self.assertEqual(row['executed_scenes'],1)
                self.assertEqual(row['unavailable_lift_outcome_trials'],1)
                self.assertEqual(row['eligible_scenes'],0)
                self.assertEqual(row['contact_without_successful_lift_trials'],0)
                self.assertEqual(row['successful_lift_trials'],0)


if __name__=='__main__':unittest.main()
