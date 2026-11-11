"""Input contracts for the independent six-object initializer and renderer."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import prepare_boya_fast6_runtime as runtime
import render_boya_fast6_observations as renderer
from prepare_egad_external_scenes import CAMERA


def protocol():
    return dict(schema=runtime.PROTOCOL_SCHEMA, object_count=6,
        objects=[dict(object_id=oid) for oid in runtime.OBJECT_IDS],
        physical_attempts_per_object=1, arm_budget_per_object=4,
        outcome_dependent_retry=False, benchmark_only=True,
        training_writeback=False, head_recipe='preactivation_selected_mse',
        source_unit_to_m=.001, object_mass_kg=.1, object_friction=.6,
        settling_policy=copy.deepcopy(runtime.POLICY),
        source_sha256={'source': 'placeholder'}, input_sha256={'input': 'placeholder'})


def manifest():
    return dict(schema='anydex_boya_external_scene_manifest_v1',
        active_object_count=6, approved_object_ids=list(runtime.OBJECT_IDS),
        objects=[dict(id=oid, object_id=oid, split='external_validation',
            scale=1, preflight_status='ready') for oid in runtime.OBJECT_IDS])


class Fast6Inputs(unittest.TestCase):
    def test_exact_six_and_no_outcome_selection(self):
        scene, plan = manifest(), protocol()
        self.assertEqual(runtime.validate_fast6_manifest(scene, plan)[1], list(runtime.OBJECT_IDS))
        for field, value in [('physical_attempts_per_object', 2), ('arm_budget_per_object', 5),
                ('outcome_dependent_retry', True), ('training_writeback', True),
                ('head_recipe', 'code100'), ('schema', 'anydex_boya_external_protocol_v1')]:
            bad = dict(plan, **{field: value})
            with self.assertRaises(ValueError):
                runtime.validate_fast6_manifest(scene, bad)

    def test_replacement_reordering_missing_and_label_rejected(self):
        for kind in ('replacement', 'reordering', 'missing', 'label', 'training_split'):
            scene = manifest()
            if kind == 'replacement':
                scene['objects'][0]['object_id'] = 'egad_A0'
            elif kind == 'reordering':
                scene['objects'].reverse()
            elif kind == 'missing':
                scene['objects'].pop()
            elif kind == 'label':
                scene['objects'][0]['label'] = 1
            else:
                scene['objects'][0]['split'] = 'train'
            with self.assertRaises(ValueError):
                runtime.validate_fast6_manifest(scene, protocol())

    def test_physics_and_scale_cannot_change(self):
        for field, value in [('source_unit_to_m', .01), ('object_mass_kg', .05),
                ('object_friction', 1.), ('settling_policy', {})]:
            with self.assertRaises(ValueError):
                runtime.validate_fast6_protocol(dict(protocol(), **{field: value}))
        scene = manifest()
        scene['objects'][0]['scale'] = .1
        with self.assertRaises(ValueError):
            runtime.validate_fast6_manifest(scene, protocol())

    def fixture(self, root):
        def write(name, value):
            path = root / name
            path.write_text(json.dumps(value))
            return str(path), runtime.sha(path)

        source, source_sha = write('source.py', 'fixture source')
        mesh, mesh_sha = write('mesh.obj', 'fixture mesh')
        urdf, urdf_sha = write('object.urdf', 'fixture urdf')
        pose, pose_sha = write('pose.json', [])
        trace, trace_sha = write('trace.json', [])
        inventory, inventory_sha = write('inventory.json', {})
        plan = protocol()
        plan.update(source_sha256={source: source_sha}, input_sha256={mesh: mesh_sha})
        plan_path, plan_sha = write('protocol.json', plan)
        scene = manifest()
        scene.update(camera=copy.deepcopy(CAMERA), source_inventory=inventory,
            source_inventory_sha256=inventory_sha, external_protocol_ref=dict(path=plan_path, sha256=plan_sha),
            settling_provenance=dict(complete=True))
        for row in scene['objects']:
            row.update(world_T_object=[[1, 0, 0, 0], [0, 1, 0, 0],
                [0, 0, 1, 0], [0, 0, 0, 1]], settling_ready=True,
                mesh_path=mesh, mesh_sha256=mesh_sha, urdf_path=urdf,
                urdf_sha256=urdf_sha, pose_path=pose, pose_sha256=pose_sha,
                convex_parts=[dict(path=mesh, sha256=mesh_sha)],
                source_input_sha256={mesh: mesh_sha},
                settling=dict(trace=trace, trace_sha256=trace_sha))
        path, unused = write('objects.json', scene)
        return Path(path), Path(mesh), scene

    def test_renderer_verifies_sources_without_isaac_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, mesh, scene = self.fixture(Path(tmp))
            loaded, rows, digest, inputs = renderer.load_manifest(path)
            self.assertEqual([r['object_id'] for r in rows], list(runtime.OBJECT_IDS))
            self.assertIn(str(mesh), inputs)
            mesh.write_text('changed source mesh')
            with self.assertRaises(ValueError):
                renderer.load_manifest(path)

    def test_renderer_refuses_subset_or_unsettled_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, unused, scene = self.fixture(Path(tmp))
            with self.assertRaises(ValueError):
                renderer.load_manifest(path, ['egad_A1'])
            scene['objects'][0]['settling_ready'] = False
            path.write_text(json.dumps(scene))
            with self.assertRaises(ValueError):
                renderer.load_manifest(path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
