"""CPU-prefix budget/selection/lifecycle regressions without real simulation."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import author_cpu_prefix_control as C
import author_cpu_prefix_audit as A


def group(n,prefix,outer,peak):
    return dict(count=n,metrics=dict(prefix_envsteps_per_s=prefix,
        startup_inclusive_envsteps_per_s=outer,prefix_intervals=dict(peak_concurrency=peak)))


POLICY=dict(minimum_both_rate_gain=1.10,tie_relative_band=.05)


class PrefixTests(unittest.TestCase):
    def test_prefix_alone_or_nominal_workers_cannot_qualify(self):
        value=A.select([group(1,10,1,1),group(4,30,.9,4),group(8,40,4,1)],POLICY)
        self.assertIsNone(value['selected_parallel_count'])
        self.assertFalse(value['full_followup_allowed'])

    def test_frozen_tie_uses_smaller_budget_and_ignores_labels(self):
        groups=[group(1,10,1,1),group(4,20,2,4),group(8,21,2.1,8)]
        value=A.select(groups,POLICY)
        self.assertEqual(value['selected_parallel_count'],4)
        groups[1]['diagnostic_success_rate']=0;groups[2]['diagnostic_success_rate']=1
        self.assertEqual(A.select(groups,POLICY),value)

    def test_clear_larger_throughput_selects_eight(self):
        value=A.select([group(1,10,1,1),group(4,20,2,4),group(8,40,3,8)],POLICY)
        self.assertEqual(value['selected_parallel_count'],8)

    def test_invalid_outer_budget_is_rejected_before_gpu_or_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/C.PROTOCOL).write_text(json.dumps(dict(schema=C.SCHEMA,prefix_ticks=100,
                retry_budget=0,total_worker_attempt_budget=14,total_integrated_environment_steps=1300,
                stage_timeout_s=900.,resource_limits={'kit':8},exclude_training=True)))
            with self.assertRaisesRegex(ValueError,'budget'):C.verify(root)

    def test_existing_start_refuses_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'RUN_STARTED.json').write_text('{}')
            with patch.object(C,'verify',return_value={}),patch.object(C.C,'run_group') as dispatch:
                with self.assertRaisesRegex(ValueError,'retried'):C.run(root)
                dispatch.assert_not_called()

    def test_first_failure_stops_later_groups_and_releases_registry(self):
        class Sampler:
            def __init__(self,root,p):self.path=root/'samples.jsonl';self.closed=False
            def start(self):self.path.write_text('')
            def close(self):self.closed=True
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/C.PROTOCOL).write_text('{}')
            p=dict(inputs={'nvidia_smi':{'path':'fixture'}},sources={},gpu_uuid='fixture',gpu_identity={},
                groups=[dict(name='cpu_1'),dict(name='cpu_4'),dict(name='cpu_8')])
            lock=root/'global.lock';registry=root/'registry.json'
            with patch.object(C,'verify',return_value=p),patch.object(C.C,'gpu_identity',return_value={}),\
                 patch.object(C.B,'verify'),patch.object(C.G,'LOCK',lock),patch.object(C.G,'REGISTRY',registry),\
                 patch.object(C.C,'ResourceSampler',Sampler),patch.object(C.C,'run_group',
                    return_value=dict(status='stopped_after_failure',name='cpu_1')) as dispatch:
                value=C.run(root)
            self.assertEqual(dispatch.call_count,1)
            self.assertEqual(value['status'],'stopped_after_failure')
            self.assertEqual(json.loads(registry.read_text())['state'],'released')
            self.assertEqual(value['new_training_labels'],0)
            with C.R.lease(lock):pass


if __name__=='__main__':unittest.main()
