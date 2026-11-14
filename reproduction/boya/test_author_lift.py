"""All-body lift labels and native shared-clock orchestration, without Kit."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from reproduction.boya import author_lift as lift
from reproduction.boya import author_transport as common


IDS = ['primary', 'other', 'third']


def state(z=.03, bottom=.001, loads=None):
    return dict(pose_xyzw=[0., 0., z, 0., 0., 0., 1.], source_bottom_m=bottom,
        finger_loads_n=[0.]*5 if loads is None else loads,
        illegal_robot_support_force_n=0., velocity=[0.]*6)


def trace(winner='other'):
    rows = []
    for i, phase in enumerate(lift.POLICY['required_phases']):
        objects = {key:state() for key in IDS}
        if phase == 'lift': objects[winner] = state(.09, .06, [.002, .002, 0., 0., 0.])
        rows.append(dict(step=i+1, time_s=(i+1)*.004, phase=phase, objects=objects))
    return rows


class LiftTests(unittest.TestCase):
    def test_any_original_body_can_succeed_without_primary_identity(self):
        result = lift.evaluate_lift(trace(), IDS, .004)
        self.assertTrue(result['task_success'])
        self.assertEqual([r['object_id'] for r in result['object_outcomes'] if r['task_success']], ['other'])
        self.assertFalse(result['objects']['primary']['task_success'])
        result2 = lift.evaluate_lift(trace('third'), list(reversed(IDS)), .004)
        self.assertEqual([r['object_id'] for r in result2['object_outcomes'] if r['task_success']], ['third'])

    def test_no_extra_half_second_gate_but_drop_and_single_side_fail(self):
        result = lift.evaluate_lift(trace(), IDS, .004)
        self.assertEqual(result['objects']['other']['trailing_observed_pickup_run_s'], .004)
        self.assertTrue(result['task_success'])
        for failure in ('drop', 'bottom', 'rise', 'thumb', 'opposing'):
            rows = trace()
            if failure == 'drop':
                rows.append(copy.deepcopy(rows[-1])); rows[-1].update(step=7, time_s=.028)
                rows[-1]['objects']['other'] = state()
            if failure == 'bottom': rows[-1]['objects']['other']['source_bottom_m'] = .0049
            if failure == 'rise': rows[-1]['objects']['other']['pose_xyzw'][2] = .0799
            if failure == 'thumb': rows[-1]['objects']['other']['finger_loads_n'][0] = .001
            if failure == 'opposing': rows[-1]['objects']['other']['finger_loads_n'][1] = .001
            with self.subTest(failure=failure):
                self.assertFalse(lift.evaluate_lift(rows, IDS, .004)['task_success'])

    def test_approach_only_or_missing_ordered_closure_cannot_label_success(self):
        for wrong in ('no_lift', 'no_close', 'wrong_order', 'release', 'repeated_close'):
            rows = trace()
            if wrong == 'no_lift': rows[-1]['phase'] = 'approach'
            if wrong == 'no_close': rows[3]['phase'] = 'approach'
            if wrong == 'wrong_order': rows[3]['phase'], rows[4]['phase'] = 'hold', 'close'
            if wrong == 'release': rows[-1]['phase'] = 'release'
            if wrong == 'repeated_close': rows[1]['phase'] = 'close'
            with self.subTest(wrong=wrong):
                self.assertFalse(lift.evaluate_lift(rows, IDS, .004)['task_success'])

    def test_trace_rejects_missing_body_nonfinite_or_misaligned_clock(self):
        for wrong in ('body', 'nan', 'clock', 'step', 'loads'):
            rows = trace()
            if wrong == 'body': del rows[-1]['objects']['primary']
            if wrong == 'nan': rows[-1]['objects']['other']['source_bottom_m'] = float('nan')
            if wrong == 'clock': rows[-1]['time_s'] += .004
            if wrong == 'step': rows[-1]['step'] += 1
            if wrong == 'loads': rows[-1]['objects']['other']['finger_loads_n'] = [.01]
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                lift.evaluate_lift(rows, IDS, .004)

    def test_full_source_bottom_uses_rotation_and_native_origin(self):
        vertices = np.array([[1., 0., 2.], [-1., 0., -2.], [0., 1., 1.]])*.01
        from scipy.spatial.transform import Rotation
        rotation = Rotation.from_euler('xyz', [31., -27., 12.], degrees=True)
        pose = np.r_[[.2, -.3, .06], rotation.as_quat()]
        expected = (rotation.as_matrix()@vertices.T).T+pose[:3]
        result = lift.source_metrics(vertices, pose)
        self.assertAlmostEqual(result['source_bottom_m'], expected[:, 2].min(), places=15)
        self.assertAlmostEqual(result['source_top_m'], expected[:, 2].max(), places=15)
        self.assertNotIn('inside_bin', result)

    def runtime(self, root):
        runtime = lift.MultiObjectLiftRuntime.__new__(lift.MultiObjectLiftRuntime)
        runtime.scene_path = root/'scene.json'; common.save(runtime.scene_path, {'bin':None})
        runtime.scene = dict(objects=[dict(object_id=i) for i in IDS], target_object_id='primary')
        runtime.output_dir = root; runtime.rows = trace(); runtime.contact_rows = []
        runtime.bodies = dict.fromkeys(IDS); runtime.steps = len(runtime.rows); runtime.dt = .004
        runtime.max_penetration = runtime.max_illegal_object_force = 0.
        runtime.collection_binding = dict(task=lift.TASK, collection_candidate_sha256='a'*64)
        runtime.commands_sha256 = 'b'*64
        return runtime

    def test_finish_requires_all_original_seven_and_all_body_validity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = dict(complete=True, task_success=False,
                gates={key:True for key in lift.ORIGINAL_GATES})
            result = self.runtime(root).finish(original)
            self.assertTrue(result['task_success']); self.assertTrue(result['label_eligible'])
            self.assertTrue(result['objects']['other']['task_success'])
            self.assertEqual(result['original_object_ids'], IDS)
            self.assertFalse(result['legacy_primary_task_success_used'])
            for gate in lift.ORIGINAL_GATES:
                bad = copy.deepcopy(original); del bad['gates'][gate]
                with self.subTest(gate=gate):
                    result = self.runtime(root).finish(bad)
                    self.assertFalse(result['task_success']); self.assertFalse(result['label_eligible'])
            for kind in ('arm_support', 'penetration', 'error', 'early_stop'):
                runtime = self.runtime(root)
                if kind == 'arm_support': runtime.max_illegal_object_force = .0011
                if kind == 'penetration': runtime.max_penetration = .0011
                result = runtime.finish(original, error='native error' if kind == 'error' else None,
                    physical_stop='push guard' if kind == 'early_stop' else None)
                with self.subTest(kind=kind):
                    self.assertFalse(result['task_success']); self.assertFalse(result['label_eligible'])
                    self.assertTrue(result['objects']['other']['observed_final_lift_success'])

    def test_complete_negative_is_eligible_but_zero_step_failure_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); runtime = self.runtime(root)
            runtime.rows[-1]['objects']['other'] = state()
            original = dict(complete=True, gates=dict.fromkeys(lift.ORIGINAL_GATES, True))
            result = runtime.finish(original)
            self.assertTrue(result['label_eligible']); self.assertFalse(result['task_success'])
            runtime = self.runtime(root); runtime.rows = []; runtime.steps = 0; runtime.bodies = {}
            result = runtime.finish({}, error='source setup failed before ready')
            self.assertFalse(result['label_eligible']); self.assertEqual(result['total_physics_steps'], 0)
            self.assertEqual(len(result['object_outcomes']), 3)

    def test_shared_clock_updates_primary_and_extras_once(self):
        counts = dict(primary=0, extras=0, writes=0, ready=0, clock=0)
        sim = SimpleNamespace(get_physics_step_count=lambda:counts['clock'])
        finished = []
        class Runtime:
            def __init__(self, *a, **kw): self.steps = 0
            factory = None
            def ready(self): counts['ready'] += 1
            def before_step(self): counts['writes'] += 1
            def after_step(self, phase): counts['extras'] += 1
            def finish(self, original, error, stop): finished.append((self.steps, error, stop))
        def original(args, **kw):
            try:
                yield dict(kind='ready', sim=sim)
                for i in range(6):
                    yield dict(kind='step', sim=sim, step=i+1)
                    counts['primary'] += 1
            finally: common.save(args.output_dir/'evaluation.json', {})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); scene = root/'scene.json'; common.save(scene, {})
            common.save(root/'plan.json', dict(approach_only=False, author_lift=dict(task=lift.TASK,
                scene_sha256=common.sha(scene)), commands_sha256='b'*64,
                author_collection_binding=dict(task=lift.TASK)))
            np.savez(root/'commands.npz', phase=np.asarray(lift.POLICY['required_phases']))
            args = SimpleNamespace(plan_dir=root, output_dir=root)
            with patch.object(common, '_backend_path'), patch.object(lift, 'MultiObjectLiftRuntime', Runtime), patch.dict(sys.modules, {
                'execute_template_arm_positive':SimpleNamespace(execution_steps=original),
                'run_template_arm_positive':SimpleNamespace(bind_hand=lambda args:None)}):
                for event in lift.execution_steps(args, scene):
                    if event['kind'] == 'step': counts['clock'] += 1
        self.assertEqual(counts, dict(primary=6, extras=6, writes=6, ready=1, clock=6))
        self.assertEqual(finished, [(6, None, None)])


if __name__ == '__main__': unittest.main()
