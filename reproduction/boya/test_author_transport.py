"""Transport geometry/label/native-buffer checks; no simulator is launched."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reproduction.boya.author_transport import (
    POLICY, MultiObjectRuntime, _waypoint_targets, args_for_plan, bin_boxes,
    decode_contacts, evaluate_transport, object_robot_loads, preclosure_pushes, save, sha, source_metrics, validate_scene, verify_pins)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.bin = dict(world_T_bin=np.eye(4).tolist(), inner_half_extents_m=[.15, .15, .1],
                        floor_thickness_m=.01, wall_thickness_m=.01, friction=.6)

    def test_source_pins_and_exact_retreat_direction(self):
        verify_pins()
        palm = np.eye(4); palm[:3, 3] = [-.7, .1, .2]
        axis = np.array([.6, 0, -.8])
        targets = _waypoint_targets(palm, axis, .1, self.bin, np.eye(4))
        np.testing.assert_allclose(targets[0][1][:3, 3], palm[:3, 3]-.004*axis, atol=1e-16)
        np.testing.assert_allclose(targets[1][1][:3, 3], palm[:3, 3]-.1*axis, atol=1e-16)
        self.assertEqual([x[0] for x in targets], ['retreat_4mm', 'retreat_pregrasp', 'transport_bin'])
        self.assertEqual(POLICY['retreat_wait_s']/POLICY['dt_s'], 25)
        np.testing.assert_array_equal(targets[2][1][:3, :3], palm[:3, :3])

    def test_bin_physical_boxes_and_full_source_containment(self):
        boxes = bin_boxes(self.bin)
        self.assertEqual(len(boxes), 5)
        self.assertAlmostEqual(boxes[0]['world_T_box'][2][3]+boxes[0]['extents_m'][2]/2, -.1)
        vertices = np.array([[-.01, -.02, -.03], [.01, .02, .03]])
        metrics = source_metrics(vertices, [0, 0, 0, 0, 0, 0, 1], self.bin)
        self.assertTrue(metrics['inside_bin'])
        metrics = source_metrics(vertices, [.145, 0, 0, 0, 0, 0, 1], self.bin)
        self.assertFalse(metrics['inside_bin'])  # reference inside, surface outside
        frame = np.eye(4); frame[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        frame[:3, 3] = [.4, -.3, .2]
        rotated = dict(self.bin, world_T_bin=frame.tolist())
        metrics = source_metrics(vertices, [.4, -.3, .2, 0, 0, 0, 1], rotated)
        self.assertTrue(metrics['inside_bin'])

    def test_any_object_label_requires_rise_release_and_final_window(self):
        n = round(POLICY['final_window_s']/POLICY['dt_s'])
        good = dict(source_bottom_m=.08, inside_bin=True, robot_load_n=0., velocity=[0.]*6,
                    finger_loads_n=[.01, .01, 0., 0., 0.], illegal_robot_support_force_n=0.)
        bad = dict(good, inside_bin=False)
        phases = ['approach', 'close', 'hold', 'retreat_4mm', 'retreat_wait',
                  'retreat_pregrasp', 'transport_bin', 'release']
        rows = [dict(phase=phase, objects={'legacy': bad, 'other': copy.deepcopy(good)}) for phase in phases]
        rows += [dict(phase='release_settle', objects={'legacy': bad, 'other': dict(good, source_bottom_m=.001)}) for _ in range(n)]
        result = evaluate_transport(rows, {'legacy': .001, 'other': .001}, POLICY['dt_s'])
        self.assertEqual(result['delivered_object_ids'], ['other'])
        for mutation in ('load', 'speed', 'containment', 'phase', 'rise', 'short',
                         'only_approach_toss', 'no_bilateral', 'arm_support', 'drop_before_release', 'missing_close'):
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(rows)
                if mutation == 'load': changed[-1]['objects']['other']['robot_load_n'] = .01
                if mutation == 'speed': changed[-1]['objects']['other']['velocity'][0] = .1
                if mutation == 'containment': changed[-1]['objects']['other']['inside_bin'] = False
                if mutation == 'phase': changed[-1]['phase'] = 'release'
                if mutation == 'rise':
                    for row in changed: row['objects']['other']['source_bottom_m'] = .01
                if mutation == 'short': changed = changed[-(n-1):]
                if mutation == 'only_approach_toss':
                    changed = [changed[0]]+changed[-n:]
                if mutation == 'no_bilateral':
                    for row in changed: row['objects']['other']['finger_loads_n'] = [.01, 0., 0., 0., 0.]
                if mutation == 'arm_support':
                    for row in changed: row['objects']['other']['illegal_robot_support_force_n'] = 1.
                if mutation == 'drop_before_release':
                    changed[6]['objects']['other']['finger_loads_n'] = [0.]*5
                if mutation == 'missing_close':
                    changed = [row for row in changed if row['phase'] != 'close']
                self.assertEqual(evaluate_transport(changed, {'legacy': .001, 'other': .001}, POLICY['dt_s'])['delivered_object_ids'], [])

    def test_native_contact_pairs_preserve_dynamic_object_and_bin_identity(self):
        bodies = ['/World/Object', '/World/AuthorScene/other']
        targets = ['/World/Robot/ff1', '/World/Obstacles/AuthorBin_floor', '/World/Object']
        count = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int32)
        start = np.array([[0, 0, 0], [0, 1, 2]], dtype=np.int32)
        data = [np.array([1., .98, .2]), np.zeros((3, 3)), np.ones((3, 3)), np.array([0., -.0001, 0.]), count, start]
        decoded = decode_contacts(bodies, targets, data, 100)
        self.assertEqual(decoded[1]['bodies'], [bodies[1], targets[1]])
        self.assertEqual(decoded[2]['bodies'], [bodies[1], targets[2]])
        for mutation in ('capacity', 'negative', 'nonfinite', 'interval', 'shape', 'geometry_shape'):
            with self.subTest(mutation=mutation):
                bad = [x.copy() for x in data]
                capacity = 100
                if mutation == 'capacity': capacity = 3
                if mutation == 'negative': bad[4][0, 0] = -1
                if mutation == 'nonfinite': bad[1][1, 0] = np.nan
                if mutation == 'interval': bad[5][1, 1] = 20
                if mutation == 'shape': bad[4] = bad[4][:1]
                if mutation == 'geometry_shape': bad[1] = np.zeros((3, 2))
                with self.assertRaises((RuntimeError, FloatingPointError)):
                    decode_contacts(bodies, targets, bad, capacity)

    def test_all_additional_dynamic_bodies_write_every_tick(self):
        class Body:
            def __init__(self): self.writes = 0
            def write_data_to_sim(self): self.writes += 1
        runtime = object.__new__(MultiObjectRuntime)
        runtime.target_id = 'compat'
        runtime.bodies = {name: Body() for name in ('compat', 'second', 'third')}
        for _ in range(7): runtime.before_step()
        self.assertEqual([b.writes for b in runtime.bodies.values()], [0, 7, 7])

    def test_distractor_arm_support_is_illegal_with_original_force_threshold(self):
        robot = ['/World/Robot/th1', '/World/Robot/ff1', '/World/Robot/forearm_link']
        contacts = [dict(bodies=['/World/Object', robot[0]], normal_force_n=.01),
                    dict(bodies=['/World/AuthorScene/other', robot[0]], normal_force_n=.02),
                    dict(bodies=['/World/AuthorScene/other', robot[1]], normal_force_n=.03),
                    dict(bodies=['/World/AuthorScene/other', robot[2]], normal_force_n=-.98)]
        target = object_robot_loads('/World/Object', contacts, robot, {'th1', 'ff1'})
        other = object_robot_loads('/World/AuthorScene/other', contacts, robot, {'th1', 'ff1'})
        self.assertEqual(target['illegal_robot_support_force_n'], 0.)
        self.assertEqual(other['illegal_robot_support_force_n'], .98)
        self.assertEqual(other['finger_loads_n'], [.02, .03, 0., 0., 0.])
        self.assertGreater(other['illegal_robot_support_force_n'], .001)

    def test_scene_binding_rejects_changed_mesh_or_mass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'mesh').mkdir(); mesh = root/'mesh/simplified.obj'; mesh.write_text('source binding')
            row = dict(object_id='a', object_root=str(root), scale=1., world_T_object=np.eye(4).tolist(),
                       mass_kg=.1, friction=.6, source_mesh=str(mesh), source_mesh_sha256=sha(mesh))
            scene = dict(schema='anydex_boya_author_scene_v1', target_object_id='a',
                         objects=[row, dict(row, object_id='b')], bin=self.bin)
            path = root/'scene.json'; save(path, scene)
            self.assertEqual(len(validate_scene(path)['objects']), 2)
            scene['objects'][1]['mass_kg'] = .2; save(path, scene)
            with self.assertRaises(ValueError): validate_scene(path)
            scene['objects'][1]['mass_kg'] = .1; save(path, scene)
            mesh.write_text('changed')
            with self.assertRaises(ValueError): validate_scene(path)

    def test_zero_step_initialization_failure_preserves_original_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'mesh').mkdir(); mesh = root/'mesh/simplified.obj'; mesh.write_text('source binding')
            row = dict(object_id='a', object_root=str(root), scale=1., world_T_object=np.eye(4).tolist(),
                       mass_kg=.1, friction=.6, source_mesh=str(mesh), source_mesh_sha256=sha(mesh))
            path = root/'scene.json'
            save(path, dict(schema='anydex_boya_author_scene_v1', target_object_id='a',
                            objects=[row, dict(row, object_id='b')], bin=self.bin))
            runtime = MultiObjectRuntime(path, root/'run')
            result = runtime.finish({}, error='original native setup exception')
            self.assertEqual(result['status'], 'infrastructure_error')
            self.assertEqual(result['error'], 'original native setup exception')
            self.assertEqual(result['total_physics_steps'], 0)
            self.assertFalse(result['label_eligible'])

    def test_exact_capture_archive_is_allowed_but_never_substitutes_asset_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root/'mesh').mkdir()
            mesh = root/'mesh/simplified.obj'; mesh.write_text('asset')
            live, archive = root/'code.py', root/'capture_code.py'
            live.write_text('old capture'); archive.write_bytes(live.read_bytes())
            row = dict(object_id='a', object_root=str(root), scale=1., world_T_object=np.eye(4).tolist(),
                       mass_kg=.1, friction=.6, source_mesh=str(mesh), source_mesh_sha256=sha(mesh))
            scene = dict(schema='anydex_boya_author_scene_v1', target_object_id='a',
                objects=[row, dict(row, object_id='b')], bin=self.bin,
                source_sha256={str(live): sha(live)}, input_sha256={str(mesh): sha(mesh)})
            path = root/'scene.json'; save(path, scene); original_sha = sha(path)
            live.write_text('new execution implementation')
            with self.assertRaises(ValueError): validate_scene(path)
            validate_scene(path, {str(live): str(archive)})
            self.assertEqual(sha(path), original_sha)
            archive.write_text('wrong historical version')
            with self.assertRaises(ValueError): validate_scene(path, {str(live): str(archive)})
            archive.write_text('old capture'); mesh.write_text('changed physical asset')
            with self.assertRaises(ValueError): validate_scene(path, {str(live): str(archive)})

    def test_preclosure_push_guard_is_any_body_and_keeps_original_threshold(self):
        states = {key: dict(pose_xyzw=[x, 0., 0., 0., 0., 0., 1.])
                  for key, x in (('compat', 0.), ('other', .031))}
        initial = {'compat': np.zeros(2), 'other': np.zeros(2)}
        self.assertEqual(preclosure_pushes(states, initial, 'approach'), ['other'])
        self.assertEqual(preclosure_pushes(states, initial, 'transport_bin'), [])
        states['other']['pose_xyzw'][0] = .03
        self.assertEqual(preclosure_pushes(states, initial, 'approach'), [])


if __name__ == '__main__':
    unittest.main()
