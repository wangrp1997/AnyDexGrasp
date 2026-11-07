"""S40 source-bound null and actual frozen-pilot expansion checks; zero physics."""
from pathlib import Path
import copy
import importlib.util
import json
import sys
import tempfile
import unittest

ROOT=Path('/home/wangrenpeng/AnyDexGrasp')
sys.path.insert(0,str(ROOT/'reproduction/boya'))
p=Path(__file__).with_name('run_s40_collection.py')
spec=importlib.util.spec_from_file_location('s40_collection_draft',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class NullContract(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);r=Path(self.tmp.name)
  self.obs=r/'observation.json';self.obs.write_text(json.dumps(dict(object_id='graspnet_005')))
  self.raw=r/'raw.npz';self.raw.write_bytes(b'bound-test-candidate-bytes')
  self.cal=r/'calibration.json';self.cal.write_text('{}')
  self.job=dict(trial_id='graspnet_005_g6_d2',object_id='graspnet_005',gesture_id=6,extra_depth_index=2,observation_json=str(self.obs),source_npz=str(self.raw))
  self.null=dict(status='no_hand_geometry_in_declared_budget',physics_steps=0,training_label=None,gesture_id=6,records=[dict(additional_depth_index=2)],source_candidates_sha256=m.sha(self.raw),source_observation_sha256=m.sha(self.obs),calibration_sha256=m.sha(self.cal))
 def test_ordinary_zero_step_geometry_null_is_not_negative(self):
  result=m.validate_null(self.null,self.job,self.cal);self.assertIsNone(result['training_label']);self.assertEqual(result['physics_steps'],0)
 def test_foreign_label_depth_gesture_or_physical_null_rejected(self):
  for update in [dict(training_label=0),dict(physics_steps=1),dict(gesture_id=7),dict(records=[dict(additional_depth_index=3)]),dict(status='unhandled_error')]:
   with self.subTest(update=update),self.assertRaises(ValueError):m.validate_null(dict(self.null,**update),self.job,self.cal)
 def test_tampered_input_and_foreign_object_rejected(self):
  self.raw.write_bytes(b'changed')
  with self.assertRaises(ValueError):m.validate_null(self.null,self.job,self.cal)
  self.raw.write_bytes(b'bound-test-candidate-bytes')
  with self.assertRaises(ValueError):m.validate_null(self.null,dict(self.job,object_id='graspnet_007'),self.cal)

class RealPilotExpansion(unittest.TestCase):
 def test_actual_pilot_reconciles_to_frozen640_with632_new(self):
  with tempfile.TemporaryDirectory()as tmp:
   oldargv=sys.argv;sys.argv=['run_s40_collection.py','--run-dir',str(Path(tmp)/'collection'),'--dry-run']
   try:args=m.parse_args()
   finally:sys.argv=oldargv
   protocol=m.freeze(args);slots=protocol['slots'];inherited=m.read(args.run_dir/'INHERITED_PILOT.json')
   self.assertEqual(len(slots),640);self.assertEqual(len({r['trial_id']for r in slots}),640)
   self.assertEqual(len(inherited['terminal_slot_ids']),8);self.assertEqual(len([s for s in slots if s['trial_id']not in inherited['terminal_slot_ids']]),632)
   self.assertEqual(len(inherited['actual_rows']),5);self.assertEqual(sum(r['physics_steps']for r in inherited['actual_rows']),10500)
   self.assertTrue(all(s['split']=='train'for s in slots))
   self.assertEqual([len(protocol['internal_mesh_groups'][x])for x in ['fit','internal_check']],[32,8])
   rows=m.read(args.run_dir/'objects.json')['objects'];self.assertEqual(len(rows),40)
   self.assertEqual(sum(r['observation_inherited_from_pilot']for r in rows),2)
   self.assertEqual(len(m.read(args.run_dir/'representation_manifest.json')['objects']),38)
   self.assertIn(m.read(args.scene_manifest)['source_inventory'],protocol['input_sha256'])
   m.verify(protocol['input_sha256']);m.verify(protocol['source_sha256']);m.verify(protocol['manifest_sha256'])
   args.plan_workers=7
   with self.assertRaises(ValueError):m.freeze(args)
 def test_real_frame_binding_rejects_foreign_observation_or_camera(self):
  row=m.read(ROOT/'outputs/s40_alignment_pilot_20261106_v1/scenes/objects.json')['objects'][0]
  receipt=m.read(Path(row['representation_dir'])/'RESULTS.json')
  m.verify_frame_binding(row,receipt)
  for mutate in ['observation','camera','input']:
   bad=copy.deepcopy(receipt)
   if mutate=='observation':bad['scene_binding']['observation_sha256']='0'*64
   if mutate=='camera':bad['scene_binding']['T_WC'][0][3]+=.01
   if mutate=='input':bad['input_files'].pop(next(iter(bad['input_files'])))
   with self.subTest(mutate=mutate),self.assertRaises(ValueError):m.verify_frame_binding(row,bad)
 def test_changed_scale_or_test_membership_rejected(self):
  scene=m.read(ROOT/'outputs/s40_alignment_pilot_20261106_v1/scenes/objects.json')
  for mutate in ['scale','split','identity']:
   changed=copy.deepcopy(scene)
   if mutate=='scale':changed['objects'][0]['scale']=.1
   if mutate=='split':changed['objects'][0]['split']='validation'
   if mutate=='identity':changed['objects'][0]['graspnet_object_id']=87
   with self.subTest(mutate=mutate),self.assertRaises(ValueError):m.validate_scene(changed)
 def test_legacy_five_actual_rows_pass_new_full_execution_inspector(self):
  from s40_execution_outcomes import inspect_execution
  for oid in ['graspnet_000','graspnet_002']:
   p=ROOT/'outputs/s40_alignment_pilot_20261106_v1/execution'/oid/'RESULTS.json';doc=m.read(p)
   result=inspect_execution(doc,[r['trial_id']for r in doc['trials']],0)
   self.assertTrue(result['complete']);self.assertEqual(result['unknown'],0)
   self.assertEqual(result['eligible_labels'],len(doc['trials']))

if __name__=='__main__':unittest.main(verbosity=2)
