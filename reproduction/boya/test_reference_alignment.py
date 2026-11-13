"""Numerical contracts for the opt-in fork-reference alignment; no GPU/physics."""
import ast
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from augmented_candidates import ROOT, augmentation_branches, merge_branches, original_numerics, recover_candidates
from audit_reference_geometry import URDFFK, axis_rotation
from run_augmented_representation import capture_next_rng


class AugmentationContracts(unittest.TestCase):
    def test_original_aggregator_generates_identical_schedule_before_forward(self):
        functions = {n.name:n for n in ast.parse((ROOT/'robot_dh3.py').read_text()).body if isinstance(n, ast.FunctionDef)}
        calls = []
        def get_grasp(*args, **kwargs):
            calls.append((kwargs['augment_mat'].copy(), kwargs.get('flip', False)))
            return None, None, None, None, []
        scope = dict(np=np, POINTCLOUD_AUGMENT_NUM=10, augment_data=original_numerics()[0],
                     get_grasp=get_grasp, get_depth=lambda _:None)
        exec(compile(ast.Module(body=[functions['get_ggarray_features']], type_ignores=[]), 'official_aggregator', 'exec'), scope)
        np.random.seed(20261108); expected = augmentation_branches(); after = np.random.get_state()
        np.random.seed(20261108); scope['get_ggarray_features'](None, None, None)
        self.assertEqual(len(calls), 11)
        for (actual, f), (reference, rf) in zip(calls, expected):
            np.testing.assert_array_equal(actual, reference); self.assertEqual(f, rf)
        self.assertEqual(np.random.get_state()[2], after[2])
        np.testing.assert_array_equal(np.random.get_state()[1], after[1])

    def test_recovery_of_translated_reflected_nontrivial_grasps(self):
        np.random.seed(47)
        r = axis_rotation([1, 2, 3], .71) @ axis_rotation([0, 1, 0], -.4)
        p = np.array([.17, -.23, .52])
        for matrix, flip in augmentation_branches():
            with self.subTest(flip=flip, matrix=matrix.tolist()):
                row = np.zeros((1, 15), np.float32); row[0, :3] = [.9, .05, .03]
                row[0, 12:] = matrix[:3, :3]@p + matrix[:3, 3]
                # Reflection of a right-handed grasp also flips local closing Y.
                row[0, 3:12] = (matrix[:3, :3]@r@np.diag([1, -1 if flip else 1, 1])).reshape(9)
                row_before, matrix_before = row.copy(), matrix.copy()
                restored = recover_candidates(row, matrix, flip)
                np.testing.assert_allclose(restored[0, 12:], p, atol=8e-8, rtol=0)
                np.testing.assert_allclose(restored[0, 3:12].reshape(3, 3), r, atol=2e-7, rtol=0)
                np.testing.assert_array_equal(restored[:, :3], row[:, :3])
                np.testing.assert_array_equal(row, row_before); np.testing.assert_array_equal(matrix, matrix_before)

    def test_reflection_parity_invalid_rotation_and_empty(self):
        empty = np.empty((0, 15), np.float32)
        self.assertEqual(recover_candidates(empty, np.eye(4), False).shape, (0, 15))
        with self.assertRaises(ValueError): recover_candidates(empty, np.eye(4), True)
        rows = np.zeros((1, 15), np.float32)
        with self.assertRaises(ValueError): recover_candidates(rows, np.eye(4), False)

    def test_merge_preserves_feature_pairing_and_source_rows_across_null_branch(self):
        branches = []
        for i, count in enumerate([0, 3, 0, 2]):
            c = np.zeros((count, 15), np.float32); f = np.zeros((count, 2261), np.float32)
            for j in range(count): c[j, 0] = f[j, 240] = 100*i+j
            branches.append((c, f))
        original = copy.deepcopy(branches); out = merge_branches(branches)
        np.testing.assert_array_equal(out['source_branch'], [1, 1, 1, 3, 3])
        np.testing.assert_array_equal(out['source_branch_row'], [0, 1, 2, 0, 1])
        np.testing.assert_array_equal(out['candidates'][:, 0], out['candidate_features'][:, 240])
        for before, after in zip(original, branches):
            for a, b in zip(before, after): np.testing.assert_array_equal(a, b)
        self.assertEqual(merge_branches([])['candidate_features'].shape, (0, 2261))
        with self.assertRaises(ValueError): merge_branches([(branches[1][0], branches[1][1][:1])])

    def test_forward_rng_advances_across_branches(self):
        expected_cpu = torch.Generator().manual_seed(8)
        expected_numpy = np.random.RandomState(9)
        torch.manual_seed(8); np.random.seed(9)
        engine = SimpleNamespace(device='cuda:0')
        with patch('torch.cuda.get_rng_state', return_value=torch.zeros(4, dtype=torch.uint8)):
            for _ in range(11):
                capture_next_rng(engine)
                # This is the legacy helper's restore at the start of a call.
                torch.set_rng_state(engine.cpu_rng); np.random.set_state(engine.numpy_rng)
                np.testing.assert_array_equal(torch.rand(3).numpy(), torch.rand(3, generator=expected_cpu).numpy())
                np.testing.assert_array_equal(np.random.random(3), expected_numpy.random(3))


class IndependentFKContracts(unittest.TestCase):
    def test_origin_order_local_axis_mimic_and_prismatic(self):
        robot = ET.fromstring('''<robot name="test">
          <link name="palm"/><link name="first"/><link name="second"/><link name="third"/>
          <joint name="one" type="revolute"><parent link="palm"/><child link="first"/>
            <origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/><axis xyz="1 0 0"/></joint>
          <joint name="two" type="revolute"><parent link="first"/><child link="second"/>
            <origin xyz="0 1 0"/><axis xyz="0 0 1"/><mimic joint="one" multiplier="0.5" offset="0.2"/></joint>
          <joint name="three" type="prismatic"><parent link="second"/><child link="third"/>
            <axis xyz="0 0 2"/></joint></robot>''')
        poses = URDFFK(robot).evaluate(dict(one=np.pi/2, three=.3))
        expected = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        np.testing.assert_allclose(poses['first'][:3, :3], expected, atol=1e-15)
        np.testing.assert_allclose(poses['second'][:3, 3], [1, 2, 4], atol=1e-15)
        np.testing.assert_allclose(poses['second'][:3, :3]@[1, 0, 0], [0, np.cos(np.pi/4+.2), np.sin(np.pi/4+.2)], atol=1e-15)
        np.testing.assert_allclose(poses['third'][:3, 3], [1.3, 2, 4], atol=1e-15)
        with self.assertRaises(KeyError): URDFFK(robot).evaluate(dict(three=.3))

    def test_relative_to_palm_cancels_wrist_motion(self):
        robot = ET.fromstring('''<robot name="test"><link name="world"/><link name="palm"/><link name="tip"/>
          <joint name="wrist" type="revolute"><parent link="world"/><child link="palm"/>
            <origin xyz="4 5 6"/><axis xyz="0 1 0"/></joint>
          <joint name="finger" type="fixed"><parent link="palm"/><child link="tip"/>
            <origin xyz="1 2 3" rpy="0.2 0.3 0.4"/><axis xyz="0 0 0"/></joint></robot>''')
        fk = URDFFK(robot)
        np.testing.assert_allclose(fk.evaluate(dict(wrist=.9))['tip'], fk.evaluate(dict(wrist=-.4))['tip'], atol=1e-14)


if __name__ == '__main__':
    unittest.main()
