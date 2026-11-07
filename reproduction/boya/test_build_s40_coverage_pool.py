import importlib.util
from pathlib import Path
import tempfile,json,hashlib,unittest
import numpy as np

p=Path('/tmp/build_s40_coverage_pool.py');s=importlib.util.spec_from_file_location('coverage_builder',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def vector(i):
 x=np.zeros(480,np.float32);x[i]=1;return x

def history(features=(),gestures=(),extras=(),counts=None):
 return m.History(np.zeros((4,20),np.int64)if counts is None else counts,np.asarray(features,np.float32).reshape(-1,480),gestures,extras)

def candidate(i,d=0,feature=None,flip=False):
 f=vector(i)if feature is None else feature
 return dict(source_candidate_index=i,if_flip=flip,raw_depth_index=d,cgr_score=1.-i*.001,feature=f,feature_sha256=m.fingerprint(f))

def admitted(row):
 return dict(ok=True,meta=dict(geometry=dict(ok=True)),arrays=dict(palm=np.eye(4),q=[0.],q_close=[1.],approach=[0.,0.,-1.]))

class CoverageTests(unittest.TestCase):
 def test_depth_strata_retain_low_quality_rare_depth_beyond_global_topk(self):
  depths=np.array([2]*20+[0,1,3,4]);raw=np.zeros((24,15),np.float32);raw[:,0]=np.linspace(1.,.01,24);raw[:,2]=depths*.01
  features=np.zeros((24,2261),np.float32)
  for i,d in enumerate(depths):features[i,i]=1;features[i,-1]=d*.01
  def canonical(row,feature,if_flip):return dict(two_fingers_pose_depth_type=int(feature[-1]*100+.1),canonical_CGR480=feature[:480],if_flip=if_flip)
  cat,a=m.candidate_catalog(raw,features,canonical,history(),1,0,10)
  self.assertEqual(a['source_rows_considered_by_raw_depth'],[1,1,2,1,1]);self.assertTrue(all(cat));self.assertLessEqual(sum(a['source_rows_considered_by_raw_depth']),10)

 def test_same_feature_gesture_extra_actual_excluded_across_indices(self):
  raw=np.zeros((5,15),np.float32);raw[:,0]=1;features=np.zeros((5,2261),np.float32)
  for i in range(5):features[i,i]=1;features[i,-1]=i*.01
  def canonical(row,feature,if_flip):return dict(two_fingers_pose_depth_type=int(feature[-1]*100+.1),canonical_CGR480=feature[:480],if_flip=if_flip)
  h=history([vector(0)],[6],[0]);cat,a=m.candidate_catalog(raw,features,canonical,h,6,0,5)
  self.assertEqual(cat[0],[]);self.assertEqual(a['excluded_previous_actual_variants_by_raw_depth'][0],2)
  cat,_=m.candidate_catalog(raw,features,canonical,h,6,1,5);self.assertEqual(len(cat[0]),2)
  cat,_=m.candidate_catalog(raw,features,canonical,h,7,0,5);self.assertEqual(len(cat[0]),2)

 def test_diversity_prefers_far_new_canonical_over_near_new(self):
  near=vector(0)+.01*vector(1);h=history([vector(0)],[1],[0])
  ordered=m.diverse_variants([candidate(1,feature=near),candidate(2,feature=vector(1))],h)
  self.assertEqual(ordered[0]['source_candidate_index'],2);self.assertTrue(ordered[0]['novel_to_any_previous_actual'])

 def test_failed_flip_does_not_erase_valid_other_flip_same_feature(self):
  a=candidate(1,flip=False);b=candidate(1,flip=True);cat=[[a,b],[],[],[],[]]
  def assess(row):return admitted(row)if row['if_flip']else dict(ok=False,meta=dict(geometry=dict(ok=False,reason='table')))
  selected,reject,checks=m.collect_balanced(cat,list(range(5)),assess)
  self.assertEqual(len(selected),1);self.assertTrue(selected[0]['meta']['if_flip']);self.assertEqual(len(reject),1)
  self.assertIsNone(reject[0]['training_label'])

 def test_all_depths_preserved_keep8_max2_and_everything_unlabelled(self):
  cat=[[candidate(d*10+i,d)for i in range(3)]for d in range(5)]
  selected,reject,a=m.collect_balanced(cat,[4,0,1,2,3],admitted)
  self.assertEqual(len(selected),8);self.assertEqual(a['selected_by_raw_depth'],[2,2,1,1,2]);self.assertLessEqual(max(a['admitted_by_raw_depth']),2)
  self.assertTrue(all(x['meta']['training_label']is None and x['meta']['physics_steps']==0 for x in selected))

 def test_all_rejections_are_null_and_hard_geometry_cap(self):
  cat=[[candidate(d*80+i,d)for i in range(80)]for d in range(5)]
  def reject(row):return dict(ok=False,meta=dict(geometry=dict(ok=False,reason='visible_scene_point_in_hand_hull')))
  selected,rejected,a=m.collect_balanced(cat,list(range(5)),reject)
  self.assertEqual(selected,[]);self.assertEqual(a['geometry_calls_by_raw_depth'],[64]*5);self.assertEqual(len(rejected),320)
  self.assertTrue(all(x['training_label']is None for x in rejected))

 def test_coverage_priority_uses_counts_and_is_stable(self):
  counts=np.zeros((4,20),np.int64)
  for depth,count in enumerate([8,2,0,7,1]):counts[0,depth*4+2]=count
  h=history(counts=counts)
  self.assertEqual(m.depth_priority(h,1,2,'abc',4),[2,4,1,3,0]);self.assertEqual(m.depth_priority(h,1,2,'abc',4),m.depth_priority(h,1,2,'abc',4))

 def test_changed_history_root_is_rejected(self):
  with tempfile.TemporaryDirectory()as temp:
   r=Path(temp);root=r/'source.json';root.write_text('{}');npz=r/'history.npz';np.savez_compressed(npz,counts=np.zeros((4,20),np.int64),features=vector(0)[None],gesture_ids=np.array([1]),extra_depth_indices=np.array([0]))
   doc=dict(schema='anydex_s40_coverage_history_v1',complete=True,source_builder_sha256=m.sha(p),root_input_sha256={str(root):m.sha(root)},history_npz=dict(path=str(npz),sha256=m.sha(npz)),actual_attempts=1)
   out=r/'history.json';out.write_text(json.dumps(doc));m.load_history(out,m.Reader());root.write_text('{"edited":true}')
   with self.assertRaisesRegex(ValueError,'SHA mismatch'):m.load_history(out,m.Reader())

 def test_bad_source_pair_or_budget_rejected(self):
  with self.assertRaises(ValueError):m.candidate_catalog(np.zeros((1,15)),np.zeros((1,1)),None,history(),1,0,1000)
  with self.assertRaises(ValueError):m.collect_balanced([[]]*5,list(range(5)),admitted,8,65)

if __name__=='__main__':unittest.main()
