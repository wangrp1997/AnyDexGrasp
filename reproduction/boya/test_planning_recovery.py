"""Boundary tests for the bounded planning intervention."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from planning_recovery import POLICY, extra_seeds, different_branches, only_home_collisions


class RecoveryContract(unittest.TestCase):
    def test_low_discrepancy_seeds_deterministic_and_bounded(self):
        names = list('abcdef')
        lower, upper = np.full(6, -6.), np.full(6, 6.)
        first = extra_seeds(names, lower, upper)
        self.assertEqual(first, extra_seeds(names, lower, upper))
        values = np.array([[r[n] for n in names] for r in first])
        self.assertEqual(values.shape, (16, 6))
        self.assertTrue(np.all(values > lower))
        self.assertTrue(np.all(values < upper))
        self.assertEqual(len(np.unique(values, axis=0)), 16)

    def test_no_baseline_or_duplicate_branch_retry_and_cap(self):
        names = list('abcdef')
        row = lambda x: dict(state=dict.fromkeys(names, x), seed_index=x)
        rows = [row(0), row(0), row(1), row(1+1e-7), row(2), row(3), row(4), row(5)]
        result = different_branches(rows, dict.fromkeys(names, 0), names)
        self.assertEqual([r['seed_index'] for r in result], [1, 2, 3, 4])

    def test_transit_allowed_only_if_all_violations_are_in_home(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p/'plan.json').write_text(json.dumps(dict(phases=[dict(name='home_to_pregrasp', end_step=800)])))
            (p/'lift_arm_geometry.json').write_text(json.dumps(dict(ok=True)))
            def permitted(samples):
                (p/'path_geometry.json').write_text(json.dumps(dict(violations=[dict(sample=s) for s in samples])))
                return only_home_collisions(p)
            self.assertTrue(permitted([50, 799]))
            self.assertFalse(permitted([49]))
            self.assertFalse(permitted([800]))
            self.assertFalse(permitted([200, 1200]))
            self.assertFalse(permitted([]))
            permitted([500])
            (p/'lift_arm_geometry.json').write_text(json.dumps(dict(ok=False)))
            self.assertFalse(only_home_collisions(p))

    def test_no_relaxed_grasp_or_physics_gate(self):
        for key in ('grasp_pose_change','object_pose_change','hand_command_change','collision_tolerance_change','curobo_used'):
            self.assertIs(POLICY[key], False)
        self.assertEqual(POLICY['physics_steps'], 0)
        self.assertEqual([POLICY[n+'_steps'] for n in ('approach','close','hold','lift')], [500,250,50,500])


if __name__ == '__main__': unittest.main()
