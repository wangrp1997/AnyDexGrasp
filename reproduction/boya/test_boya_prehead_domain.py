"""Pure tests, including direct numeric comparison to pinned author AST."""
import ast
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from boya_prehead_domain import apply_boya_domain, released_dh3_camera_mask

SOURCE = Path('/home/wangrenpeng/AnyDexGrasp/robot_dh3.py')


def rows(n=3, cols=15):
    a = np.zeros((n, cols), dtype=np.float64)
    a[:, 0] = np.arange(n) + .1
    a[:, 1] = .05
    a[:, 2] = .005
    a[:, 3:12] = np.eye(3).reshape(1, 9)
    return a


def config():
    return dict(world_workspace_aabb_m=[[-1., -1., -1.], [1., 1., 1.]],
                approach_world_cone=dict(axis_world=[1., 0., 0.], cos_threshold=.9),
                raw_width_bounds_m=[.04, .06])


PROVENANCE = dict(raw_candidates=dict(path='/frozen/raw.npz', sha256='a' * 64),
                  observation=dict(path='/frozen/observation.json', sha256='b' * 64))


def apply(a, T=None, c=None, provenance=None):
    return apply_boya_domain(a, np.eye(4) if T is None else T,
                            config() if c is None else c,
                            source_provenance=PROVENANCE if provenance is None else provenance)


class BoyaPreheadDomainTests(unittest.TestCase):
    def test_world_transform_uses_approach_column_and_no_translation_for_axis(self):
        a = rows(1)
        a[0, 12:15] = [1., 2., 3.]
        T = np.array([[0., 0., 1., 10.], [0., 1., 0., 20.],
                      [-1., 0., 0., 30.], [0., 0., 0., 1.]])
        c = config()
        c['world_workspace_aabb_m'] = [[12., 21., 28.], [14., 23., 30.]]
        c['approach_world_cone'] = dict(axis_world=[0., 0., -1.], cos_threshold=.99)
        result = apply(a, T, c)
        np.testing.assert_allclose(result.world_points_m, [[13., 22., 29.]])
        np.testing.assert_allclose(result.world_approach_axes, [[0., 0., -1.]])
        self.assertEqual(result.keep_indices.tolist(), [0])

    def test_source_indices_keep_feature_pairing_and_do_not_rank_by_score(self):
        a = rows(5)
        a[1, 1] = .08
        a[3, 12] = 2.
        a[4, 3:12] = np.diag([-1., -1., 1.]).reshape(-1)
        old = a.copy()
        features = np.arange(35).reshape(5, 7)
        result = apply(a)
        self.assertEqual(result.keep_indices.tolist(), [0, 2])
        np.testing.assert_array_equal(features[result.keep_indices], features[[0, 2]])
        np.testing.assert_array_equal(a, old)
        self.assertEqual(result.receipt['counts'], dict(input=5, raw_width_pass=4,
            world_workspace_pass=4, world_approach_pass=4, after_width=4,
            after_width_workspace=3, joint_pass=2))

    def test_strict_width_boundaries(self):
        a = rows(4)
        a[:, 1] = [.04, .05, .06, .055]
        self.assertEqual(apply(a).keep_indices.tolist(), [1, 3])

    def test_strict_workspace_boundaries_on_each_axis(self):
        a = rows(7)
        a[1, 12] = -1.; a[2, 12] = 1.
        a[3, 13] = -1.; a[4, 13] = 1.
        a[5, 14] = -1.; a[6, 14] = 1.
        self.assertEqual(apply(a).keep_indices.tolist(), [0])

    def test_strict_cone_boundary_rejects_equal_dot(self):
        a = rows(2)
        a[1, 3:12] = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]).reshape(-1)
        c = config(); c['approach_world_cone']['cos_threshold'] = 0.
        self.assertEqual(apply(a, c=c).keep_indices.tolist(), [0])

    def test_boya_has_no_author_camera_rectangle_or_dh3_width_defaults(self):
        a = rows(1); a[:, 1] = .20; a[:, 12] = 3.
        T = np.eye(4); T[0, 3] = -3.
        c = config(); c['raw_width_bounds_m'] = [.15, .25]
        self.assertEqual(apply(a, T=T, c=c).keep_indices.tolist(), [0])
        self.assertFalse(released_dh3_camera_mask(a, .045, .099).any())

    def test_n16_trailing_metadata_preserves_xyz_columns_and_hash(self):
        a = rows(1, 16); a[0, 15] = 12345.
        result = apply(a)
        self.assertEqual(result.keep_indices.tolist(), [0])
        self.assertEqual(result.receipt['raw_input']['shape'], [1, 16])
        self.assertEqual(result.receipt['raw_layout'], 'score,width,depth,R9,XYZ,unused_tail')
        np.testing.assert_array_equal(result.world_points_m, [[0., 0., 0.]])
        b = a.copy(); b[0, 15] = 12346.
        self.assertNotEqual(result.receipt['raw_input'], apply(b).receipt['raw_input'])

    def test_n15_empty_input_produces_zero_counts_and_valid_indices(self):
        result = apply(rows(0))
        self.assertEqual(result.keep_indices.dtype, np.int64)
        self.assertEqual(result.keep_indices.tolist(), [])
        self.assertEqual(result.receipt['counts']['joint_pass'], 0)
        self.assertEqual(result.world_approach_axes.shape, (0, 3))

    def test_wrong_layout_nonfinite_improper_rotation_and_depth_rejected(self):
        bad = [rows(2, 17), np.zeros((3, 14)), rows(1)]
        bad[-1][0, 3] = np.nan
        reflection = rows(1); reflection[0, 3:12] = np.diag([-1., 1., 1.]).reshape(-1)
        scaled = rows(1); scaled[0, 3] = 1.001
        negative_depth = rows(1); negative_depth[0, 2] = -.01
        for a in bad + [reflection, scaled, negative_depth]:
            with self.subTest(shape=a.shape), self.assertRaises(ValueError): apply(a)

    def test_camera_transform_rigidity_and_shape_rejected(self):
        bad = [np.eye(3), np.eye(4), np.eye(4), np.eye(4), np.eye(4)]
        bad[1][3, 0] = .1; bad[2][0, 0] = -1.; bad[3][0, 0] = 1.001; bad[4][0, 3] = np.inf
        for T in bad:
            with self.subTest(T=T), self.assertRaises(ValueError): apply(rows(1), T)

    def test_config_requires_all_explicit_values_and_rejects_typos(self):
        for key in config():
            c = config(); del c[key]
            with self.subTest(missing=key), self.assertRaises(ValueError): apply(rows(1), c=c)
        c = config(); c['author_camera_rectangle'] = True
        with self.assertRaises(ValueError): apply(rows(1), c=c)

    def test_invalid_config_bounds_cone_or_cosine_rejected(self):
        bad = []
        for bounds in ([[0., 0., 0.], [0., 1., 1.]], [[0., 0., 0.], [-1., 1., 1.]]):
            c = config(); c['world_workspace_aabb_m'] = bounds; bad.append(c)
        for width in ([-.01, .06], [.06, .04], [.04, .04], [.04, float('nan')]):
            c = config(); c['raw_width_bounds_m'] = width; bad.append(c)
        for axis in ([0., 0., 0.], [2., 0., 0.], [1., 0.]):
            c = config(); c['approach_world_cone']['axis_world'] = axis; bad.append(c)
        for threshold in (1.1, -1.1, True, float('nan')):
            c = config(); c['approach_world_cone']['cos_threshold'] = threshold; bad.append(c)
        for c in bad:
            with self.subTest(config=c), self.assertRaises(ValueError): apply(rows(1), c=c)

    def test_provenance_and_receipt_are_json_copy_and_hashes_bind_input(self):
        provenance = copy.deepcopy(PROVENANCE)
        a = rows(2); T = np.eye(4); c = config()
        result = apply(a, T, c, provenance)
        provenance['raw_candidates']['path'] = '/changed'
        c['raw_width_bounds_m'][0] = .03
        self.assertEqual(result.receipt['source_provenance'], PROVENANCE)
        self.assertEqual(result.receipt['config']['raw_width_bounds_m'], [.04, .06])
        json.dumps(result.receipt, allow_nan=False)
        b = a.copy(); b[0, 0] += 1.
        self.assertNotEqual(result.receipt['raw_input'], apply(b).receipt['raw_input'])
        T[0, 3] = .01
        self.assertNotEqual(result.receipt['input_T_WC'], apply(a, T=T).receipt['input_T_WC'])
        self.assertNotEqual(result.receipt['config_sha256'], apply(a, c=c).receipt['config_sha256'])
        for provenance in ({}, {'bad': float('nan')}):
            with self.assertRaises(ValueError): apply(a, provenance=provenance)

    def test_dh3_reference_matches_actual_author_ast_for_float32_and_float64(self):
        tree = ast.parse(SOURCE.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_grasp')
        nodes = [n for n in function.body if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id in ('mask', 'workspace_mask')
                 and 'preds[' in ast.unparse(n.value)]
        self.assertEqual(len(nodes), 2)
        code = compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec')
        rng = np.random.default_rng(20261107)
        for dtype in (np.float32, np.float64):
            a = rng.normal(size=(3000, 16)).astype(dtype)
            a[:, 1] = rng.uniform(0., .2, len(a))
            a[:, 9] = rng.uniform(.8, 1., len(a))
            a[:, 12:14] = rng.uniform(-.4, .4, (len(a), 2))
            # Exact boundaries for each original predicate.
            a[:8, 1] = [.045, .099, .05, .05, .05, .05, .05, .05]
            a[:8, 9] = [1., 1., .9, 1., 1., 1., 1., 1.]
            a[:8, 12] = [0., 0., 0., -.25, .25, 0., 0., 0.]
            a[:8, 13] = [0., 0., 0., 0., 0., -.205, .03, 0.]
            for minwidth in (.045, .01):
                scope = dict(preds=a, MAX_GRASP_WIDTH=.099, MIN_GRASP_WIDTH=minwidth)
                exec(code, scope)
                expected = scope['workspace_mask'] & scope['mask']
                for cols in (15, 16):
                    np.testing.assert_array_equal(released_dh3_camera_mask(a[:, :cols], minwidth, .099), expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
