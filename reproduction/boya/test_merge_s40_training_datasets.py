"""Small end-to-end merger fixtures: no models, training, physics or Kit."""
from pathlib import Path
import copy,hashlib,importlib.util,json,tempfile,unittest
import numpy as np
P=Path(__file__).with_name('merge_s40_training_datasets.py');S=importlib.util.spec_from_file_location('s40_merge',P);m=importlib.util.module_from_spec(S);S.loader.exec_module(m)
def save(p,d):Path(p).write_text(json.dumps(d,allow_nan=False)+'\n')

class MergeContracts(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  self.objects=[]
  for index,oid in enumerate(sorted(m.S40_IDS)):
   mesh=hashlib.sha256(f'original-mesh-{oid}'.encode()).hexdigest();split='train'if index<32 else'internal_check'
   self.objects.append(dict(object_id=f'graspnet_{oid:03d}',graspnet_object_id=oid,mesh_sha256=mesh,source_visual_mesh_sha256=mesh,outer_split='train',split=split))
  self.groups={s:sorted(o['mesh_sha256']for o in self.objects if o['split']==s)for s in('train','internal_check')}

 def source(self,name):
  root=self.root/name;root.mkdir();protocol=root/'protocol.json';save(protocol,dict(unit_test_source=name))
  samples=[];actual=[];files={}
  for index,g in enumerate(m.GESTURES):
   obj=self.objects[index];tid=f'same_local_g{g}_d0';native=root/f'execution_{g}';native.mkdir()
   trace=native/'trace.npz';np.savez_compressed(trace,states=np.asarray([[g,ord(name[-1])]],dtype=np.float32))
   native_eval=native/'evaluation.json';save(native_eval,dict(unit_test_only=True,source=name,gesture=g))
   commands=root/f'commands_{g}.npz';commands.write_bytes((name+str(g)).encode());command_sha=m.sha(commands)
   feature=np.arange(480,dtype=np.float32)+g+ord(name[-1]);binding_sha=hashlib.sha256((name+tid).encode()).hexdigest()
   b=dict(multifinger_pose_finger_type=g,two_fingers_pose_depth_type=2,multifinger_pose_depth_type=0,canonical_CGR480=feature.tolist(),canonical_CGR480_identity=m.array_identity(feature),trial_binding_sha256=binding_sha)
   evaluation=root/f'bound_eval_{g}.json';label=int(name=='old')
   save(evaluation,dict(complete=True,label_eligible=True,task_success=bool(label),total_physics_steps=2,trial_binding_sha256=binding_sha,commands_sha256=command_sha,trace_sha256=m.sha(trace),original_native_evaluation_sha256=m.sha(native_eval)))
   record=root/f'record_{g}.json';save(record,dict(binding=b,label_eligible=True,result=label,execution_artifacts=dict(evaluation_json=str(evaluation),evaluation_sha256=m.sha(evaluation),commands_npz=str(commands),commands_sha256=command_sha)))
   sample=dict(trial_id=tid,object_id=obj['object_id'],gesture_id=g,mesh_sha256=obj['mesh_sha256'],split='train',label=label,target_index=8,record_json=str(record),record_sha256=m.sha(record))
   samples.append(sample);actual.append(dict(trial_id=tid,object_id=obj['object_id'],gesture_id=g,label_eligible=True,label=label,physics_steps=2,trial_binding_sha256=binding_sha,commands_sha256=command_sha,execution_dir=str(native)))
   file=root/f'gesture_{g}.npz'
   np.savez_compressed(file,features=feature[None],labels=np.asarray([label],dtype=np.int64),target_index=np.asarray([8],dtype=np.int64),gesture_ids=np.asarray([g],dtype=np.int64),mesh_sha256=m.uarray([obj['mesh_sha256']]),trial_id=m.uarray([tid]),object_id=m.uarray([obj['object_id']]),split=m.uarray(['train']))
   files[str(g)]=dict(path=str(file),sha256=m.sha(file),count=1)
  final=root/'RESULTS.json';save(final,dict(complete=True,input_sha256_unchanged=True,source_sha256_unchanged=True,trials=actual))
  manifest=root/'manifest.json';doc=dict(schema='anydex_s40_decision_dataset_v1',complete=True,collection_complete=True,benchmark_only=False,gesture_ids=list(m.GESTURES),objects=copy.deepcopy(self.objects),mesh_groups=copy.deepcopy(self.groups),sample_count=4,samples=samples,gesture_files=files,input_sha256={str(protocol):m.sha(protocol)},source_collection=dict(protocol=str(protocol),protocol_sha256=m.sha(protocol),final_results=str(final),final_results_sha256=m.sha(final)))
  save(manifest,doc);return manifest

 def test_end_to_end_preserves_bytes_and_namespaces_local_ids(self):
  sources=[self.source('old'),self.source('new')];result=m.merge(sources,self.root/'merged')
  self.assertEqual(result['sample_count'],8);self.assertEqual((result['positive'],result['negative']),(4,4))
  self.assertEqual(result['mesh_groups'],self.groups);self.assertTrue(all(o['outer_split']=='train'for o in result['objects']))
  self.assertEqual(len({r['trial_id']for r in result['samples']}),8)
  for g in m.GESTURES:
   with np.load(result['gesture_files'][str(g)]['path'],allow_pickle=False)as z:
    self.assertEqual(z['features'].dtype,np.float32);self.assertEqual(z['labels'].dtype,np.int64)
    self.assertEqual(z['features'].shape,(2,480));self.assertEqual(z['local_trial_id'][0],z['local_trial_id'][1]);self.assertNotEqual(z['trial_id'][0],z['trial_id'][1])
    self.assertEqual(set(z['source_collection_manifest']),set(map(str,sources)))
    for i,p in enumerate(sources):
     source=m.read(p)
     with np.load(source['gesture_files'][str(g)]['path'],allow_pickle=False)as original:self.assertEqual(z['features'][i].tobytes(),original['features'][0].tobytes())
   self.assertEqual(result['gesture_files'][str(g)]['target_coverage']['8'],dict(count=2,positive=1,negative=1))
  self.assertEqual(result['cost']['new_training_steps'],0)

 def test_duplicate_physical_trace_is_rejected_instead_of_dropped(self):
  row=dict(trial_id='one',source_trace_sha256='a'*64,source_trace='/tmp/one_trace')
  other=dict(row,trial_id='different_protocol_local',source_trace='/tmp/copied_trace')
  with self.assertRaisesRegex(ValueError,'trace SHA'):m.reject_duplicate_actuals([(row,None),(other,None)])
  other=dict(row,trial_id='different_protocol_local',source_trace_sha256='b'*64)
  with self.assertRaisesRegex(ValueError,'source path'):m.reject_duplicate_actuals([(row,None),(other,None)])

 def test_group_or_original_object_authority_mismatch_rejected(self):
  p=self.source('old');doc=m.read(p);doc['objects'][0]['outer_split']='test'
  with self.assertRaises(ValueError):m.authority(doc)
  doc=m.read(p);doc['mesh_groups']['internal_check'][0]=doc['mesh_groups']['train'][0]
  with self.assertRaises(ValueError):m.authority(doc)

 def test_null_or_changed_label_is_not_accepted(self):
  p=self.source('old');doc=m.read(p);record=Path(doc['samples'][0]['record_json']);r=m.read(record);r['result']=None;r['label_eligible']=False;save(record,r)
  doc['samples'][0]['record_sha256']=m.sha(record);save(p,doc)
  with self.assertRaisesRegex(ValueError,'Null or synthetic'):m.read_source(p,m.Frozen())

 def test_input_change_and_output_overwrite_rejected(self):
  old=self.source('old');new=self.source('new');out=self.root/'existing';out.mkdir()
  with self.assertRaisesRegex(ValueError,'refusing overwrite'):m.merge([old,new],out)
  doc=m.read(old);protocol=Path(doc['source_collection']['protocol']);protocol.write_text('changed')
  with self.assertRaisesRegex(ValueError,'SHA mismatch'):m.read_source(old,m.Frozen())

if __name__=='__main__':unittest.main(verbosity=2)
