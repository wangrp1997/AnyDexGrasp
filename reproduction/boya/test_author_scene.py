"""Scene provenance, source geometry and task-coordinate checks without Isaac."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import trimesh

import author_scene as scene


class SceneTests(unittest.TestCase):
    def test_actual_sources_form_common_scene_and_reject_changes(self):
        root = scene.ROOT
        with tempfile.TemporaryDirectory() as tmp:
            spec = scene.build(
                root/'outputs/s40_stable_supplement_20261107_v2/settling/objects.json',
                root/'outputs/s40_stable_supplement_20261107_v2/collection/native_preprocessing/SUMMARY.json',
                root/'reproduction/boya/backend.collection.json', Path(tmp)/'frozen')
            bounds = []
            for row in spec['objects']:
                mesh = trimesh.load_mesh(row['source_mesh'], process=False)
                mesh.apply_transform(row['world_T_object'])
                self.assertGreater(mesh.bounds[0, 2], -.002)
                bounds.append(mesh.bounds)
            for i in range(len(bounds)):
                for j in range(i):
                    self.assertFalse(np.all(np.minimum(bounds[i][1], bounds[j][1]) >
                                            np.maximum(bounds[i][0], bounds[j][0])))
            self.assertLess(np.ptp([row['world_T_object'][0][3] for row in spec['objects']]), .30)
            altered = copy.deepcopy(spec)
            altered['objects'][0]['source_mesh_sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'source geometry'):
                scene.validate(altered)
            altered = copy.deepcopy(spec)
            altered['objects'][1]['collision_refinement_sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'refinement'):
                scene.validate(altered)

    def test_bin_cavity_world_geometry_and_invalid_rotation(self):
        T = np.eye(4)
        T[:3, 3] = [-.4, -.25, .1]
        spec = dict(world_T_bin=T.tolist(), inner_half_extents_m=[.15, .15, .1],
                    wall_thickness_m=.01, floor_thickness_m=.01)
        boxes = scene.bin_boxes(spec)
        self.assertEqual(len(boxes), 5)
        lower, upper = T[:3, 3]-[.15, .15, .1], T[:3, 3]+[.15, .15, .1]
        for name, center, size in boxes:
            center, size = np.asarray(center), np.asarray(size)
            if name == 'floor':
                self.assertAlmostEqual((center+size/2)[2], lower[2])
            elif name.startswith('x_'):
                inside_x = (center+size/2)[0] if name.endswith('minus') else (center-size/2)[0]
                self.assertAlmostEqual(inside_x, lower[0] if name.endswith('minus') else upper[0])
            else:
                inside_y = (center+size/2)[1] if name.endswith('minus') else (center-size/2)[1]
                self.assertAlmostEqual(inside_y, lower[1] if name.endswith('minus') else upper[1])
            self.assertRegex(name, r'^[A-Za-z_][A-Za-z0-9_]*$')
        spec['world_T_bin'][0][0] = -1
        with self.assertRaisesRegex(ValueError, 'SE3'):
            scene.bin_boxes(spec)


if __name__ == '__main__':
    unittest.main()
