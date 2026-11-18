import json
from pathlib import Path
import tempfile
import unittest

import author_collection_batch as B
from run_author_mjx_expansion import coverage


class ObjectCoverageTests(unittest.TestCase):
    def test_untouched_distractor_is_not_a_negative_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            catalog = root/'catalog.json'
            B.save(catalog,dict(objects={k:{} for k in ('a','b','c')},
                split_object_ids=dict(fit=['a','b','c'],internal=[])))
            B.save(root/'PROTOCOL.json',dict(inputs=dict(expansion_catalog=B.binding(catalog)),
                rows=[dict(index=0,source_object_ids=['a','b','c'],scene={'path':'source'})]))
            directory = root/'scenes/00'
            trace = directory/'execution/TRACE.jsonl'
            trace.parent.mkdir(parents=True)
            state = dict(objects={k:dict(finger_loads_n=[1 if k in ('a','b') else 0]) for k in ('a','b','c')},
                contacts=[dict(bodies=['/World/Robot/thdistal','/World/a']),
                          dict(bodies=['/World/b','/World/Robot/ffdistal']),
                          dict(bodies=['/World/ground','/World/c'])])
            trace.write_text(json.dumps(state)+'\n')
            B.save(directory/'RECORD.json',dict(eligible=True))
            B.save(directory/'execution/RESULTS.json',dict(trace=B.binding(trace),evaluation=dict(
                object_outcomes=[dict(object_id=k,task_success=k=='a') for k in ('a','b','c')])))
            result = coverage(root)['objects']
            self.assertEqual(result['a']['successful_lift_trials'],1)
            self.assertEqual(result['b']['contact_without_successful_lift_trials'],1)
            self.assertEqual(result['c']['observed_robot_contact_trials'],0)
            self.assertEqual(result['c']['contact_without_successful_lift_trials'],0)


if __name__=='__main__':
    unittest.main()
