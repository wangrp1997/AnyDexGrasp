"""Contract checks for the independent S40 renderer; never starts Isaac/Kit."""
from pathlib import Path
import ast
import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest

SPEC=importlib.util.spec_from_file_location('render_s40_observations',Path(__file__).with_name('render_s40_observations.py'))
renderer=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(renderer)


class S40RendererContract(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.manifest=self.root/'scenes.json'
        self.inventory=self.root/'inventory.json';self.inventory.write_text('{"count":40}\n')
        self.frozen=dict(schema=renderer.MANIFEST_SCHEMA,active_object_count=40,scale=1,
            source_active_list=str(self.inventory),source_active_list_sha256=renderer.sha(self.inventory),
            camera=dict(width=1280,height=720,fx=640.,fy=640.,eye_world_m=[-.75,-.2,.5],
                target_world_m=[-.75,.1,.06],up_world=[0.,0.,1.],depth_range_m=[.2,.65],
                focal_length=24.,horizontal_aperture=48.,clipping_range_m=[.1,1.]),objects=[])
        for index in sorted(renderer.S40_IDS):
            oid=f'graspnet_{index:03d}';directory=self.root/oid;directory.mkdir()
            row=dict(id=oid,object_id=oid,object_root=oid,scale=1,
                selected_source_pose_index=0,world_T_object=[[1.,0.,0.,-.75],[0.,1.,0.,.1],[0.,0.,1.,.02],[0.,0.,0.,1.]],
                mass=.1,friction=.6)
            for name,content in [('mesh','source triangles'),('urdf','source URDF'),('pose','source pose')]:
                path=directory/f'{name}.txt';path.write_text(content+' '+oid)
                row[name+'_path']=str(path.relative_to(self.root));row[name+'_sha256']=renderer.sha(path)
            self.frozen['objects'].append(row)
        self.write()

    def tearDown(self):self.temp.cleanup()
    def write(self):self.manifest.write_text(json.dumps(self.frozen,allow_nan=False))

    def test_full_manifest_can_render_two_selected_ids_at_native_scale(self):
        frozen,rows,digest,inputs=renderer.load_manifest(self.manifest,['002','graspnet_000'])
        self.assertEqual([r['object_id']for r in rows],['graspnet_000','graspnet_002'])
        self.assertEqual(len(frozen['objects']),40)
        self.assertEqual(digest,renderer.sha(self.manifest))
        self.assertEqual(len(inputs),121)
        self.assertTrue(all(r['scale']==1 for r in rows))
        self.assertTrue(all(r['object_mass_kg']==.1 and r['object_friction']==.6 for r in rows))
        self.assertTrue(all(Path(r['mesh_path']).is_absolute()for r in rows))
        self.assertEqual(rows[0]['source_graspnet_object_id'],0)

    def test_input_cannot_be_reduced_or_use_old_dataset_scale(self):
        baseline=copy.deepcopy(self.frozen)
        cases=[]
        changed=copy.deepcopy(baseline);changed['objects']=changed['objects'][:2];cases.append(changed)
        changed=copy.deepcopy(baseline);changed['scale']=.1;cases.append(changed)
        changed=copy.deepcopy(baseline);changed['scale']=True;cases.append(changed)
        changed=copy.deepcopy(baseline);changed['active_object_count']=4330;cases.append(changed)
        changed=copy.deepcopy(baseline);changed['schema']='old-renderer';cases.append(changed)
        changed=copy.deepcopy(baseline);changed['objects'][0]['scale']=.1;cases.append(changed)
        for value in cases:
            with self.subTest(schema=value['schema'],count=len(value['objects']),scale=value['scale']):
                self.frozen=value;self.write()
                with self.assertRaises(ValueError):renderer.load_manifest(self.manifest,['000'])

    def test_full_source_union_and_duplicates_are_checked(self):
        self.frozen['objects'][0]['object_id']='graspnet_001';self.write()
        with self.assertRaisesRegex(ValueError,'source S40 union'):renderer.load_manifest(self.manifest)
        self.frozen['objects'][0]['object_id']='graspnet_002';self.write()
        with self.assertRaises(ValueError):renderer.load_manifest(self.manifest)

    def test_unknown_empty_and_duplicate_subset_rejected(self):
        for value in ([],['999'],['000','graspnet_000'],['000,002,'],['../graspnet_000']):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):renderer.load_manifest(self.manifest,value)
        _,rows,_,_=renderer.load_manifest(self.manifest,['000,002'])
        self.assertEqual(len(rows),2)

    def test_unselected_source_mutation_is_still_rejected(self):
        row=self.frozen['objects'][-1]
        (self.root/row['mesh_path']).write_text('changed source')
        with self.assertRaisesRegex(ValueError,'Frozen source changed'):renderer.load_manifest(self.manifest,['000'])

    def test_manifest_and_source_inventory_mutation_rejected(self):
        _,_,digest,inputs=renderer.load_manifest(self.manifest,['000'])
        self.inventory.write_text('modified inventory')
        with self.assertRaisesRegex(ValueError,'Frozen source changed'):
            renderer.verify_frozen_inputs(self.manifest,digest,inputs)
        self.manifest.write_text(self.manifest.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'Frozen scene manifest changed'):
            renderer.verify_frozen_inputs(self.manifest,digest,inputs)

    def test_rigid_pose_physical_alias_and_camera_guards(self):
        baseline=copy.deepcopy(self.frozen)
        bad=copy.deepcopy(baseline);bad['objects'][0]['world_T_object'][0][0]=2
        self.frozen=bad;self.write()
        with self.assertRaisesRegex(ValueError,'rigid transform'):renderer.load_manifest(self.manifest)
        bad=copy.deepcopy(baseline);bad['objects'][0]['object_mass_kg']=.2
        self.frozen=bad;self.write()
        with self.assertRaisesRegex(ValueError,'Conflicting physical'):renderer.load_manifest(self.manifest)
        bad=copy.deepcopy(baseline);bad['camera']['horizontal_aperture']=24.
        self.frozen=bad;self.write()
        with self.assertRaisesRegex(ValueError,'intrinsics'):renderer.load_manifest(self.manifest)

    def test_output_refuses_overwrite_and_copies_full_manifest(self):
        _,_,digest,_=renderer.load_manifest(self.manifest,['000','002'])
        out=self.root/'rendered'
        renderer.prepare_output(self.manifest,out,digest)
        self.assertEqual(renderer.sha(out/'objects.json'),digest)
        self.assertEqual(len(json.loads((out/'objects.json').read_text())['objects']),40)
        sentinel=out/'sentinel';sentinel.write_text('keep')
        with self.assertRaises(FileExistsError):renderer.prepare_output(self.manifest,out,digest)
        self.assertEqual(sentinel.read_text(),'keep')

    def test_validate_only_cli_does_not_import_or_launch_isaac(self):
        out=self.root/'not-created'
        result=subprocess.run([sys.executable,'-B',str(Path(renderer.__file__).resolve()),
            '--objects-json',str(self.manifest),'--output-dir',str(out),
            '--object-ids','000','002','--validate-only'],text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse(out.exists())
        self.assertIn('S40_RENDER_INPUTS_VALID',result.stdout)
        self.assertIn('"isaac_started": false',result.stdout)
        self.assertNotIn('isaaclab',sys.modules)

    def test_camera_and_source_ray_helpers_preserve_frozen_pilot(self):
        reference=Path('/home/wangrenpeng/AnyDexGrasp/reproduction/boya/render_pilot_observations.py')
        self.assertEqual(renderer.sha(reference),renderer.READONLY_PILOT_SOURCE_SHA256)
        actual={node.name:node for node in ast.parse(Path(renderer.__file__).read_text()).body if isinstance(node,ast.FunctionDef)}
        expected={node.name:node for node in ast.parse(reference.read_text()).body if isinstance(node,ast.FunctionDef)}
        for name in ('look_at_ros','ray_diagnostic'):
            self.assertEqual(ast.dump(actual[name],include_attributes=False),ast.dump(expected[name],include_attributes=False))
        # Preserve runtime authority and source first-hit checks. Static contract
        # only; this deliberately does not pretend to validate a native camera.
        text=Path(renderer.__file__).read_text()
        for marker in ("camera.data.pos_w","camera.data.quat_w_ros","if pose_error>2e-6:",
                "if max_error>1e-7:","if not ray['diagnostic_pass']:"):
            self.assertIn(marker,text)


if __name__=='__main__':unittest.main(verbosity=2)
