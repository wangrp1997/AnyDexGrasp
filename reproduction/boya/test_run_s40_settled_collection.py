"""Zero-physics fresh collection wiring tests; no models/Kit/executor launched."""
from pathlib import Path
import argparse,copy,importlib.util,json,tempfile,unittest
import numpy as np
from scipy.spatial.transform import Rotation

P=Path(__file__).with_name('run_s40_settled_collection.py')
S=importlib.util.spec_from_file_location('settled_collection',P);m=importlib.util.module_from_spec(S);S.loader.exec_module(m)
OLD=m.ROOT/'outputs/s40_collection_20261107_v1'
DATA=Path('/tmp/s40_training_data_export_validation_20261107_v2/manifest.json')

class FreshWiring(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.scene=m.read(OLD/'objects.json')
  results=self.root/'settling_results.json'
  proof=self.root/'settling_source.txt';proof.write_text('unit-test source receipt, no physical stability claim')
  measured=[]
  for row in self.scene['objects']:
   T=np.asarray(row['world_T_object'],float);q=Rotation.from_matrix(T[:3,:3]).as_quat();pose=np.r_[T[:3,3],q]
   trace=self.root/(row['object_id']+'_trace.npz');np.savez_compressed(trace,pose_xyzw=np.repeat(pose[None],752,axis=0),velocity=np.zeros((752,6)))
   record=dict(object_id=row['object_id'],ready=True,world_T_object=T.tolist(),measured_settled_pose_xyzw=pose.tolist(),measured_settled_velocity=[0.]*6,
    settle_window=dict(stable=True),final_window=dict(stable=True),cold_recreation_verified=True,trace=str(trace),trace_sha256=m.sha(trace),settle_steps=1,recreation_steps=750,physics_steps=751)
   posefile=self.root/(row['object_id']+'_pose.json');m.save(posefile,[[*pose[:3],q[3],*q[:3]]])
   row.update(settling_ready=True,settling=record,world_pose_xyzw=pose.tolist(),pose_path=str(posefile),pose_sha256=m.sha(posefile),selected_source_pose_index=0)
   measured.append(record)
  m.save(results,dict(complete=True,ready_objects=40,objects=measured,errors=[],unit_test_only=True))
  self.scene['settling_provenance']=dict(complete=True,ready_objects=40,results_json=str(results),results_sha256=m.sha(results),
   input_sha256={str(proof):m.sha(proof)},source_sha256={str(proof):m.sha(proof)},entry_boundary='independent cold reset; zero velocity')
  self.manifest=self.root/'settled_scene.json';m.save(self.manifest,self.scene)
  builder=self.root/'coverage.py';builder.write_text('# fake CLI source; never executed in tests\n')
  self.args=argparse.Namespace(run_dir=self.root/'fresh',scene_manifest=self.manifest,previous_protocol=OLD/'protocol.json',previous_dataset_manifest=DATA,
   previous_collection_results=OLD/'execution/RESULTS.json',native_assets_run=OLD,candidate_builder=builder,
   calibration=m.ROOT/'outputs/boya_adapter_preparation_20261105_v2/calibration.json',backend_config=m.ROOT/'outputs/boya_pilot8_20261105_v1/backend_pilot.json',
   checkpoint=Path('/mnt/public/datasets/AnyDexGrasp/logs/model/checkpoint.tar.18'),native_deps=m.ROOT/'outputs/native_deps',device='cuda:0',geometry_workers=4,plan_workers=8,execution_workers=4,native_workers=4)

 def test_all40_ready_stability_receipt_is_required(self):
  refs=m.settling_binding(self.scene);self.assertIn(self.scene['settling_provenance']['results_json'],refs)
  for kind in('incomplete','one_unready','changed_proof'):
   d=copy.deepcopy(self.scene)
   if kind=='incomplete':d['settling_provenance']['complete']=False
   elif kind=='one_unready':d['objects'][0]['settling_ready']=False
   else:d['settling_provenance']['results_sha256']='0'*64
   with self.subTest(kind=kind):
    with self.assertRaises(ValueError):m.settling_binding(d)

 def test_measured_transform_trace_and_wxyz_pose_file_are_bound(self):
  refs=m.settling_binding(self.scene);row=self.scene['objects'][0]
  self.assertIn(row['settling']['trace'],refs);self.assertIn(row['pose_path'],refs)
  changed=copy.deepcopy(self.scene);changed['objects'][0]['world_T_object'][0][3]+=.01
  with self.assertRaisesRegex(ValueError,'T differ'):m.settling_binding(changed)
  changed=copy.deepcopy(self.scene);changed['objects'][0]['settling']['measured_settled_velocity'][0]=.1
  with self.assertRaisesRegex(ValueError,'record differs'):m.settling_binding(changed)
  pose=Path(row['pose_path']);value=m.read(pose);value[0][0]+=.01;m.save(pose,value)
  row['pose_sha256']=m.sha(pose)
  with self.assertRaisesRegex(ValueError,'pose file does not reproduce'):m.settling_binding(self.scene)

 def test_fresh_freeze_preserves32_8_and_has_zero_inherited_trials(self):
  p=m.freeze(self.args);old=m.read(OLD/'protocol.json')
  self.assertEqual(p['internal_mesh_groups'],old['internal_mesh_groups'])
  self.assertEqual(p['slot_count'],640);self.assertEqual(len({j['trial_id']for j in p['slots']}),640)
  self.assertEqual(p['inherited_actual_trials'],0);self.assertEqual(p['inherited_slot_count'],0);self.assertEqual(p['inherited_physics_steps'],0)
  self.assertFalse((self.args.run_dir/'INHERITED_PILOT.json').exists())
  objects=m.read(self.args.run_dir/'objects.json')['objects'];self.assertEqual(len(objects),40)
  self.assertTrue(all(not r['observation_inherited_from_pilot']for r in objects))
  self.assertTrue(all(Path(r['observation_json']).is_relative_to(self.args.run_dir/'observations')for r in objects))
  self.assertEqual(len(m.read(self.args.run_dir/'representation_manifest.json')['objects']),40)
  self.assertIn(self.scene['settling_provenance']['results_json'],p['input_sha256'])
  self.assertIn(str(self.args.previous_protocol),p['input_sha256'])
  m.verify(p['input_sha256']);m.verify(p['source_sha256']);m.verify(p['manifest_sha256'])
  self.args.plan_workers=7
  with self.assertRaises(ValueError):m.freeze(self.args)

 def test_candidate_cli_has_bounded_history_coverage_and_new_pose_sources(self):
  cfg=dict(candidate_builder='new_builder.py',calibration='cal.json',coverage_history='coverage/history.json')
  job=dict(observation_json='new_observation.json',gesture_id=7,extra_depth_index=3,candidate_dir='fresh_pool')
  argv=m.candidate_argv(cfg,job,dict(representation_dir='new_frame'))
  for key,value in [('--candidate-limit','1000'),('--keep','8'),('--geometry-checks-per-depth','64'),('--coverage-target-per-output','4'),('--coverage-history','coverage/history.json'),('--flip-policy','both')]:
   self.assertEqual(argv[argv.index(key)+1],value)
  self.assertNotIn('build_collection_pool.py',argv)

 def test_generic_arm_preparation_is_reused_directly(self):
  self.assertIs(m.Pipeline.planning,m.OriginalPipeline.planning)
  self.assertIs(m.Pipeline.stage,m.OriginalPipeline.stage)

 def test_all_null_execution_has_no_old_five_or10500_offsets(self):
  run=self.root/'zero_actual';run.mkdir();objects=[dict(object_id=r['object_id'])for r in self.scene['objects']]
  slots=[dict(trial_id=m.job_id(r['object_id'],g,d),object_id=r['object_id'],gesture_id=g,extra_depth_index=d,split='train')for r in objects for g in m.GESTURES for d in m.DEPTHS]
  m.save(run/'objects.json',dict(objects=objects));m.save(run/'protocol.json',{})
  (run/'planning').mkdir();m.save(run/'planning/queue_manifest.json',dict(complete=True,trials=[],nulls=slots))
  pipeline=m.Pipeline(run,dict(config=dict(execution_workers=2),slot_count=640,slots=slots,input_sha256={},source_sha256={},manifest_sha256={}))
  def forbidden(*a,**k):raise AssertionError('Test must never launch a subprocess or physics')
  pipeline.command=forbidden;pipeline.publish=lambda status:None
  summary,artifacts=pipeline.execution();result=m.read(run/'execution/RESULTS.json')
  self.assertTrue(result['complete']);self.assertEqual(result['actual_grasps'],0);self.assertEqual(result['physical_steps'],0)
  self.assertEqual(result['inherited_actual_grasps'],0);self.assertEqual(result['inherited_physics_steps'],0)
  self.assertEqual(result['new_CGR_forwards'],40);self.assertEqual(result['inherited_CGR_forwards'],0)
  self.assertEqual(summary['eligible_labels'],0);self.assertEqual(summary['new_physical_steps'],0)
  self.assertEqual((run/'execution/all_records.jsonl').read_text(),'')

if __name__=='__main__':unittest.main(verbosity=2)
