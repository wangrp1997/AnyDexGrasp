"""S40 data export guards: no training/model/physics/Kit imports."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
import numpy as np

P=Path(__file__).with_name('prepare_s40_training_data.py')
SPEC=importlib.util.spec_from_file_location('prepare_s40_training_data',P)
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)

class TrainingDataContracts(unittest.TestCase):
 def setUp(self):
  self.objects=[dict(object_id=f'graspnet_{i:03d}',graspnet_object_id=i,split='train',scale=1,
   source_visual_mesh_sha256=hashlib.sha256(f'original-{i}'.encode()).hexdigest(),
   mesh_sha256=hashlib.sha256(f'converted-{i}'.encode()).hexdigest())for i in sorted(m.S40_IDS)]
  rows=[dict(object_id=o['object_id'],mesh_sha256=o['source_visual_mesh_sha256'])for o in self.objects]
  self.declared=dict(fit=rows[:32],internal_check=rows[32:])

 def test_predeclared_original_visual_mesh_split_and_outer_train(self):
  mapping,groups=m.split_plan(self.objects,self.declared)
  self.assertEqual([len(groups[k])for k in('train','internal_check')],[32,8])
  self.assertTrue(set(groups['train']).isdisjoint(groups['internal_check']))
  self.assertTrue(all(v['outer_split']=='train'for v in mapping.values()))
  self.assertEqual(mapping[self.objects[0]['object_id']]['mesh_sha256'],self.objects[0]['source_visual_mesh_sha256'])
  self.assertNotEqual(mapping[self.objects[0]['object_id']]['mesh_sha256'],self.objects[0]['mesh_sha256'])

 def test_old_dgn_external_objects_rejected(self):
  changed=copy.deepcopy(self.objects);changed[0]['split']='validation'
  with self.assertRaises(ValueError):m.split_plan(changed,self.declared)
  with self.assertRaises(ValueError):m.split_plan(self.objects+copy.deepcopy(self.objects[:10]),self.declared)

 def test_mesh_split_cannot_change_identity_or_leak(self):
  for update in ('duplicate','converted_hash','wrong_count'):
   d=copy.deepcopy(self.declared)
   if update=='duplicate':d['internal_check'][0]=copy.deepcopy(d['fit'][0])
   elif update=='converted_hash':d['fit'][0]['mesh_sha256']=self.objects[0]['mesh_sha256']
   else:d['fit']=d['fit'][:-1]
   with self.subTest(update=update):
    with self.assertRaises(ValueError):m.split_plan(self.objects,d)

 def ledger(self):
  slots=[dict(trial_id=f"{o['object_id']}_g{g}_d{d}",object_id=o['object_id'],split='train',gesture_id=g,extra_depth_index=d)
   for o in self.objects for g in m.GESTURES for d in range(4)]
  protocol=dict(schema='anydex_boya_s40_collection_v1',slot_count=640,gesture_ids=list(m.GESTURES),extra_depth_indices=[0,1,2,3],slots=slots)
  row=dict(slots[0],physics_steps=2100,label=1,label_eligible=True)
  queue=dict(complete=True,errors=[],inputs_unchanged=True,trials=[slots[0]],nulls=slots[1:])
  final=dict(schema='anydex_boya_s40_collection_results_v1',complete=True,status='complete_with_null',input_sha256_unchanged=True,
   source_sha256_unchanged=True,physical_cost_complete=True,unexecuted_ready_trial_ids=[],unknown_execution_trial_ids=[],
   trials=[row],terminal_certificates={},terminal_nulls=0,eligible_labels=1,actual_grasps=1,positives=1,negatives=0,physical_steps=2100,preparation_nulls=slots[1:])
  exports={slots[0]['trial_id']:dict(result=1)}
  return protocol,queue,final,exports

 def test_exact_eligible_identity_partition(self):
  p,q,f,e=self.ledger();slots,ready,actual,eligible,terminal=m.reconcile(p,q,f,e)
  self.assertEqual(len(slots),640);self.assertEqual(len(eligible),1);self.assertEqual(len(terminal),0)
  e['invented']=dict(result=0)
  with self.assertRaisesRegex(ValueError,'Exports'):m.reconcile(p,q,f,e)

 def test_terminal_null_cannot_be_exported_as_zero(self):
  p,q,f,e=self.ledger();row=f['trials'][0];tid=row['trial_id'];row.update(label=None,label_eligible=False)
  f.update(terminal_nulls=1,eligible_labels=0,positives=0,terminal_certificates={tid:dict(schema='anydex_native_task_stop_certificate_v1',certified=True,outcome='known_task_terminated_null',trial_id=tid)})
  with self.assertRaisesRegex(ValueError,'Exports'):m.reconcile(p,q,f,e)
  *_,eligible,terminal=m.reconcile(p,q,f,{})
  self.assertEqual(eligible,set());self.assertEqual(terminal,{tid})

 def test_duplicate_actual_and_unknown_final_rejected(self):
  p,q,f,e=self.ledger();f['trials'].append(copy.deepcopy(f['trials'][0]))
  with self.assertRaisesRegex(ValueError,'Repeated actual'):m.reconcile(p,q,f,e)
  p,q,f,e=self.ledger();f['unknown_execution_trial_ids']=['unknown']
  with self.assertRaises(ValueError):m.reconcile(p,q,f,e)

 def test_all_twenty_observed_indices_and_bad_depths(self):
  targets=[]
  for raw in range(5):
   for extra in range(4):
    b=dict(multifinger_pose_finger_type=6,two_fingers_pose_depth_type=raw,multifinger_pose_depth_type=extra)
    targets.append(m.observed_target(b,dict(gesture_id=6,extra_depth_index=extra))[-1])
  self.assertEqual(targets,list(range(20)))
  for raw in (-1,5,True):
   b=dict(multifinger_pose_finger_type=6,two_fingers_pose_depth_type=raw,multifinger_pose_depth_type=0)
   with self.assertRaises(ValueError):m.observed_target(b,dict(gesture_id=6,extra_depth_index=0))

 def test_unobserved_outputs_are_reported_without_imputation(self):
  rows=[dict(gesture_id=6,target_index=19,label=1,mesh_sha256='m1',object_id='o1',split='train'),dict(gesture_id=6,target_index=19,label=0,mesh_sha256='m2',object_id='o2',split='internal_check')]
  d=m.coverage(rows,6);self.assertEqual(d['count'],2);self.assertEqual(d['unobserved_targets'],list(range(19)))
  self.assertEqual(d['target_coverage']['19'],dict(count=2,positive=1,negative=1))

 def test_frozen_file_change_and_existing_output_rejected(self):
  with tempfile.TemporaryDirectory()as temp:
   p=Path(temp)/'source';p.write_text('original');f=m.FrozenFiles();f.bind(p);p.write_text('changed!')
   with self.assertRaisesRegex(ValueError,'Frozen input changed'):f.verify()
   with self.assertRaisesRegex(ValueError,'overwrite'):m.prepare(type('Args',(),dict(collection_run=Path(temp)/'absent',output_dir=Path(temp)))())

 def test_actual_original_canonical_and_collect_record_replay(self):
  root=Path('/home/wangrenpeng/AnyDexGrasp/outputs/s40_collection_20261107_v1')
  exported=json.loads((root/'execution/all_records.jsonl').read_text().splitlines()[0]);binding=exported['binding'];artifact=exported['execution_artifacts']
  helper=m.import_helper(m.ROOT/'reproduction/boya/trial_record.py','test_actual_s40_record')
  bound=m.FrozenFiles();feature=m.CanonicalValidator(helper,bound.bind).feature(binding)
  self.assertEqual(feature.dtype,np.float32);self.assertEqual(feature.shape,(480,))
  replay=helper.collect_record(binding,artifact['evaluation_json'],artifact['commands_npz'])
  self.assertTrue(all(exported[k]==v for k,v in replay.items()))
  changed=copy.deepcopy(binding);a=np.asarray(changed['canonical_CGR480'],dtype=np.float32);a[0]=np.nextafter(a[0],np.float32(np.inf));changed['canonical_CGR480']=a.tolist();changed['canonical_CGR480_identity']=helper.array_ref(a)
  with self.assertRaisesRegex(ValueError,'original AST'):m.CanonicalValidator(helper,bound.bind).feature(changed)

if __name__=='__main__':unittest.main(verbosity=2)
