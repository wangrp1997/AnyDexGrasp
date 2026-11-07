"""S40-only split/provenance loader guards. No optimizer, CUDA, or physics."""
from pathlib import Path
import copy,importlib.util,json,tempfile,unittest
import numpy as np
source=Path(__file__).with_name('train_s40_decision.py')
spec=importlib.util.spec_from_file_location('s40_trainer',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class LoaderChecks(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.ids=sorted(m.S40_IDS)
  self.objects=[dict(object_id=f'graspnet_{oid:03d}',mesh_sha256=f'{i:064x}',outer_split='train')for i,oid in enumerate(self.ids)]
  self.manifest=dict(schema='anydex_s40_decision_dataset_v1',complete=True,collection_complete=True,benchmark_only=False,
    mesh_groups={'train':[x['mesh_sha256']for x in self.objects[:32]],'internal_check':[x['mesh_sha256']for x in self.objects[32:]]},
    objects=self.objects,gesture_files={},input_sha256={},sample_count=16)
  self.arrays={}
  for g in [1,6,7,8]:
   a=dict(features=np.zeros((4,480),np.float32),labels=np.array([0,1,0,1],np.int64),
    target_index=np.array([0,4,12,17],np.int64),gesture_ids=np.full(4,g,np.int64),
    mesh_sha256=np.array([f'{i:064x}'for i in [0,1,32,33]]),
    trial_id=np.array([f'g{g}_{i}'for i in range(4)]),
    object_id=np.array([self.objects[i]['object_id']for i in [0,1,32,33]]),
    split=np.array(['train','train','internal_check','internal_check']))
   self.arrays[g]=a
  self.write()
 def write(self):
  for g,a in self.arrays.items():
   p=self.root/f'g{g}.npz';np.savez(p,**a)
   self.manifest['gesture_files'][str(g)]=dict(path=str(p),sha256=m.sha(p),count=4)
  self.path=self.root/'manifest.json';self.path.write_text(json.dumps(self.manifest))
 def test_original40_outer_train_32_8_internal_groups_load(self):
  manifest,data,inputs=m.load_dataset(self.path);self.assertEqual(len(data),4)
  self.assertEqual(len(data[1]['labels']),4);self.assertEqual(len(inputs),5)
 def test_fake_external_objects_or_mesh_leak_rejected(self):
  for mutate in ['external','leak']:
   with self.subTest(mutate=mutate):
    old=copy.deepcopy(self.manifest)
    if mutate=='external':self.manifest['objects'][0]['outer_split']='test'
    else:self.manifest['mesh_groups']['internal_check'][0]=self.manifest['mesh_groups']['train'][0]
    self.write()
    with self.assertRaises(ValueError):m.load_dataset(self.path)
    self.manifest=old
 def test_no_null_bool_target20_or_foreign_asterisk_split(self):
  for key,value in [('labels',np.array([0,1,-1,1],np.int64)),
                    ('labels',np.array([True,False,True,False])),
                    ('target_index',np.array([0,4,12,20],np.int64)),
                    ('split',np.array(['internal_check','train','internal_check','internal_check']))]:
   with self.subTest(key=key,value=value):
    old=self.arrays[1][key];self.arrays[1][key]=value;self.write()
    with self.assertRaises(ValueError):m.load_dataset(self.path)
    self.arrays[1][key]=old
 def test_tampered_npz_and_duplicate_actual_trial_rejected(self):
  p=self.root/'g1.npz';p.write_bytes(p.read_bytes()+b'changed')
  with self.assertRaises(ValueError):m.load_dataset(self.path)
  self.arrays[6]['trial_id'][0]=self.arrays[1]['trial_id'][0];self.write()
  with self.assertRaises(ValueError):m.load_dataset(self.path)

if __name__=='__main__':unittest.main(verbosity=2)
