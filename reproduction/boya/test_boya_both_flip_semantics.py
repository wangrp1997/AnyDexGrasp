"""No models/physics: original functions + saved planning bindings only."""
from pathlib import Path
import ast,collections,copy,json,sys,unittest
import numpy as np
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');BOYA=ROOT/'reproduction/boya'
sys.path.insert(0,str(BOYA))
from trial_record import canonical_candidate, official_functions, array_ref
FUNCTIONS=official_functions(ROOT/'robot_dh3.py')
GESTURES=[1,6,7,8]
CALLS=[]
def observed_canonical(*args,**kwargs):
 CALLS.append(kwargs['if_flip']);return canonical_candidate(*args,**kwargs)
def make_queries():
 p=BOYA/'deploy_score_first.py';tree=ast.parse(p.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='make_queries')
 scope=dict(np=np,ROOT=ROOT,GESTURES=GESTURES,official_functions=official_functions,canonical_candidate=observed_canonical,array_ref=array_ref)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),scope)
 return scope['make_queries']
MAKE=make_queries()
def fixture():
 raw=np.zeros((2,15),np.float32);raw[:,0]=[.9,.8];raw[:,1]=.05;raw[:,2]=.02;raw[0,3:12]=np.eye(3).reshape(-1);raw[1,3:12]=np.diag([1.,-1.,-1.]).reshape(-1)
 f=np.tile(np.arange(2261,dtype=np.float32),(2,1));f[:,-2]=7.;f[:,-1]=.02
 return raw,f
class BothFlipSemantics(unittest.TestCase):
 def test_each_raw_exactly_false_true_once_each_then_four_gestures(self):
  raw,f=fixture();CALLS.clear();q,x,indices=MAKE(raw,f,dict(cgr_limit=2),'both')
  self.assertEqual(CALLS,[False,True,False,True]);self.assertEqual(indices,[0,1]);self.assertEqual(len(q),16)
  self.assertEqual(len({(a['source_candidate_index'],a['if_flip'],a['gesture_id'])for a in q}),16)
  self.assertEqual([(a['source_candidate_index'],a['if_flip'])for a in q[::4]],[(0,False),(0,True),(1,False),(1,True)])
 def test_official_remains_one_policy_choice_and_both_are_explicit(self):
  raw,f=fixture();CALLS.clear();q,_,_=MAKE(raw,f,dict(cgr_limit=2),'official')
  self.assertEqual(CALLS,[None,None]);self.assertEqual([a['if_flip']for a in q[::4]],[True,False])
 def test_cgr_matches_independent_half_swap_then_angle_roll(self):
  raw,f=fixture()
  for idx in range(2):
   for flip in [False,True]:
    c=f[idx,240:720].copy()
    if flip:c=np.concatenate([c[120:240],c[:120],c[360:480],c[240:360]])
    expected=np.concatenate([np.roll(c[:240],-35),np.roll(c[240:],-35)])
    actual=canonical_candidate(raw[idx],f[idx],FUNCTIONS,if_flip=flip)
    np.testing.assert_array_equal(actual['canonical_CGR480'],expected)
    R=raw[idx,3:12].reshape(3,3).copy()
    if flip:R[:,1:3]*=-1
    np.testing.assert_array_equal(actual['candidate_gg17_after_flip'][4:13].reshape(3,3),R)
 def test_repeated_calls_preserve_unmodified_source_and_do_not_accumulate_flips(self):
  raw,f=fixture();oldraw=raw.copy();oldf=f.copy()
  q1,x1,_=MAKE(raw,f,dict(cgr_limit=2),'both');q2,x2,_=MAKE(raw,f,dict(cgr_limit=2),'both')
  self.assertEqual(q1,q2);np.testing.assert_array_equal(x1,x2);np.testing.assert_array_equal(raw,oldraw);np.testing.assert_array_equal(f,oldf)
 def test_double_canonicalization_is_not_semantically_idempotent(self):
  raw,f=fixture()
  for flip in [False,True]:
   once=canonical_candidate(raw[0],f[0],FUNCTIONS,if_flip=flip)
   bad=f[0].copy();bad[240:720]=once['canonical_CGR480']
   twice=canonical_candidate(raw[0],bad,FUNCTIONS,if_flip=flip)
   self.assertFalse(np.array_equal(once['canonical_CGR480'],twice['canonical_CGR480']))
 def test_depth_and_unappended_feature_contract_reject_mismatch(self):
  raw,f=fixture();bad=f[0].copy();bad[-1]=.01
  with self.assertRaises(ValueError):canonical_candidate(raw[0],bad,FUNCTIONS,if_flip=False)
  with self.assertRaises(ValueError):canonical_candidate(raw[0],np.r_[f[0],False],FUNCTIONS,if_flip=False)
 def test_equal_depth_feature_swap_requires_source_byte_pairing_guard(self):
  raw,f=fixture();f[1,240]+=1000
  expected=array_ref(f[0]);swapped=f[1].copy()
  self.assertNotEqual(array_ref(swapped),expected)
  # Shape/depth alone cannot certify pairing; retain source NPZ/index/byte SHA.
  wrong=canonical_candidate(raw[0],swapped,FUNCTIONS,if_flip=False)
  right=canonical_candidate(raw[0],f[0],FUNCTIONS,if_flip=False)
  self.assertNotEqual(array_ref(wrong['canonical_CGR480']),array_ref(right['canonical_CGR480']))
 def test_all27_saved_ready_bindings_match_both_queries_exactly(self):
  base=ROOT/'outputs/s40_stable_supplement_20261107_v2/collection_resume_v3'
  queue=json.loads((base/'planning/queue_manifest.json').read_text())
  ids={p.parent.name for p in (ROOT/'outputs/s40_domain_diagnostic_20261107_v1').glob('graspnet_*/DOMAIN.json')}
  ready=[r for r in queue['trials']if r['object_id']in ids];self.assertEqual(len(ready),27)
  groups=collections.defaultdict(list)
  for row in ready:groups[json.loads(Path(row['binding_json']).read_text())['source_npz']].append(row)
  matched=[]
  for path,rows in groups.items():
   bmap={r['trial_id']:json.loads(Path(r['binding_json']).read_text())for r in rows}
   ix=sorted({b['source_row_index']for b in bmap.values()})
   with np.load(path,allow_pickle=False)as z:raw=z['candidates'][ix].copy();features=z['candidate_features'][ix].copy()
   q,x,_=MAKE(raw,features,dict(cgr_limit=len(ix)),'both');querymap={(ix[a['source_candidate_index']],a['gesture_id'],a['if_flip']):(a,v)for a,v in zip(q,x)}
   for row in rows:
    b=bmap[row['trial_id']];a,v=querymap[b['source_row_index'],row['gesture_id'],b['if_flip']]
    self.assertEqual(a['canonical_CGR480_identity'],b['canonical_CGR480_identity'])
    np.testing.assert_array_equal(v,np.asarray(b['canonical_CGR480'],np.float32))
    self.assertEqual(a['raw_depth_index'],b['two_fingers_pose_depth_type'])
    self.assertEqual(a['official_policy_if_flip'],b['official_policy_if_flip'])
    self.assertEqual(row['extra_depth_index'],b['multifinger_pose_depth_type'])
    matched.append(dict(trial_id=row['trial_id'],source_row_index=b['source_row_index'],if_flip=b['if_flip'],extra_depth_index=row['extra_depth_index'],canonical480_exact=True))
  Path('/tmp/boya_both_flip_binding_matches_20261107.json').write_text(json.dumps(dict(schema='anydex_boya_both_flip_semantics_check_v1',ready_bindings_matched=len(matched),matches=matched,labels_read=False,physics_steps=0,model_forwards=0),indent=2)+'\n')
if __name__=='__main__':unittest.main(verbosity=2)
