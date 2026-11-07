"""Independent collection renderer contracts; no Kit/model/physics process."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import shutil
import subprocess
import sys
import unittest


def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj

HERE=Path(__file__).resolve().parent
renderer=module('s40_collection_renderer_under_test',HERE/'render_s40_collection_observations.py')
base_path=HERE/'test_s40_renderer.py'
if not base_path.is_file():base_path=HERE/'test_render_s40_observations.py'
if not base_path.is_file():
    base_path=Path('/home/wangrenpeng/AnyDexGrasp/reproduction/boya/test_render_s40_observations.py')
base=module('_s40_original_guard_tests',base_path)
base.renderer=renderer


class CollectionRendererContracts(base.S40RendererContract):
    def setUp(self):
        super().setUp()
        self.frozen['source_inventory']=str(self.inventory)
        self.frozen['source_inventory_sha256']=renderer.sha(self.inventory)
        self.write()

    def test_full_manifest_can_render_two_selected_ids_at_native_scale(self):
        frozen,rows,digest,inputs=renderer.load_manifest(self.manifest,['002','graspnet_000'])
        self.assertEqual([r['object_id']for r in rows],['graspnet_000','graspnet_002'])
        self.assertEqual(len(frozen['objects']),40)
        self.assertEqual(digest,renderer.sha(self.manifest))
        self.assertEqual(len(inputs),123)  # 120 assets + inventory + two sources
        self.assertEqual(inputs[str(self.inventory)],renderer.sha(self.inventory))
        self.assertEqual(inputs[str(renderer.PILOT_PATH)],renderer.PILOT_RENDERER_SHA256)
        self.assertEqual(inputs[str(Path(renderer.__file__).resolve())],renderer.sha(renderer.__file__))
        self.assertTrue(all(r['scale']==1 for r in rows))
        self.assertTrue(all(r['object_mass_kg']==.1 and r['object_friction']==.6 for r in rows))
        self.assertTrue(all(Path(r['mesh_path']).is_absolute()for r in rows))

    def test_camera_and_source_ray_helpers_preserve_frozen_pilot(self):
        self.assertEqual(renderer.sha(renderer.PILOT_PATH),renderer.PILOT_RENDERER_SHA256)
        self.assertIs(renderer.look_at_ros,renderer._pilot.look_at_ros)
        self.assertIs(renderer.ray_diagnostic,renderer._pilot.ray_diagnostic)
        actual=Path(renderer.__file__).read_text();expected=renderer.PILOT_PATH.read_text()
        # Full mesh creation, native camera/raster capture and ray checks retained.
        begin="        camera_spec=frozen['camera'];"
        end="            report=dict("
        self.assertEqual(actual[actual.index(begin):actual.index(end)],expected[expected.index(begin):expected.index(end)])
        for marker in ("camera.data.pos_w","camera.data.quat_w_ros","if pose_error>2e-6:",
                "if max_error>1e-7:","if not ray['diagnostic_pass']:"):
            self.assertIn(marker,actual)

    def test_inventory_path_and_hash_are_mandatory(self):
        baseline=copy.deepcopy(self.frozen)
        for key in ('source_inventory','source_inventory_sha256'):
            self.frozen=copy.deepcopy(baseline);del self.frozen[key];self.write()
            with self.subTest(key=key):
                with self.assertRaises(ValueError):renderer.load_manifest(self.manifest,['000'])

    def test_inventory_compatibility_digest_mismatch_rejected(self):
        self.frozen['source_inventory_sha256']='0'*64;self.write()
        with self.assertRaisesRegex(ValueError,'conflicts'):renderer.load_manifest(self.manifest,['000'])

    def test_inventory_path_alias_must_resolve_to_same_file(self):
        alias=self.root/'copied_inventory.json';shutil.copyfile(self.inventory,alias)
        self.frozen['source_active_list']=str(alias);self.write()
        with self.assertRaisesRegex(ValueError,'same inventory path'):renderer.load_manifest(self.manifest,['000'])

    def test_inventory_frozen_without_source_active_path(self):
        del self.frozen['source_active_list'];self.write()
        _,_,digest,inputs=renderer.load_manifest(self.manifest,['000'])
        self.assertIn(str(self.inventory),inputs)
        self.inventory.write_text('changed')
        with self.assertRaisesRegex(ValueError,'Frozen source changed'):
            renderer.verify_frozen_inputs(self.manifest,digest,inputs)

    def test_source_digest_pin_rejects_modified_dependency_before_kit(self):
        sandbox=self.root/'renderer_copy';sandbox.mkdir()
        entry=sandbox/'render_s40_collection_observations.py';shutil.copyfile(renderer.__file__,entry)
        dep=sandbox/'render_s40_observations.py';shutil.copyfile(renderer.PILOT_PATH,dep)
        dep.write_text(dep.read_text()+'\n')
        result=subprocess.run([sys.executable,'-B',str(entry),'--objects-json',str(self.manifest),
            '--output-dir',str(self.root/'uncreated'),'--validate-only'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Frozen S40 pilot renderer dependency changed',result.stderr)
        self.assertFalse((self.root/'uncreated').exists())


if __name__=='__main__':unittest.main(verbosity=2)
