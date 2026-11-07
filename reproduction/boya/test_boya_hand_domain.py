from pathlib import Path
import sys,unittest
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent));sys.path.insert(0,'/home/wangrenpeng/AnyDexGrasp/reproduction/boya')
from boya_hand_domain import calibrated_opening,plane_minimum,make_hand_queries,expand_hand_queries
from deploy_score_first import PROFILES
class HandDomainContracts(unittest.TestCase):
 def setUp(self):
  self.vertices=np.asarray([[x,y,z]for x in [-.01,.01]for y in [-.01,.01]for z in [-.01,.01]])
  class Hand:
   def frames(self,q):return {'body':np.eye(4)},None
  self.assets=SimpleNamespace(hand=Hand(),meshes={'body':SimpleNamespace(vertices=self.vertices)})
  self.families={g:[dict(slot_id=f'g{g}:s{i}',reference_width_m=w,independent_q=[g,i],palm_T_reference=np.eye(4).tolist())for i,w in enumerate([.03,.08])]for g in [1,6,7,8]}
  self.policy=dict(schema='anydex_boya_hand_domain_policy_v1',world_workspace_aabb_m=[[-1,-1,0],[1,1,1]],
   table_height_world_m=0.,table_clearance_m=.0005,pregrasp_distance_m=.1,raw_width_bounds_m=[0.,.099],width_expansion=1.7,
   nearest_slot_tolerance_m=.002,width_clamp=True,extra_depth_m=[0.,.01,.02,.03],flip_policy='both',source_sha256={'fixture':'source'})
 def arrays(self):
  raw=np.zeros((2,15),np.float32);raw[:,0]=[.9,.8];raw[:,1]=.01;raw[:,2]=.02;raw[:,3:12]=np.eye(3).reshape(-1);raw[:,14]=.2
  features=np.zeros((2,2261),np.float32);features[:,-1]=raw[:,2]
  for i in range(2):features[i,240:720]=np.arange(480)+i*1000
  return raw,features
 def test_original_clamp_and_tie_rule_preserved(self):
  slots=self.families[1]
  chosen,meta=calibrated_opening(.001,slots);self.assertEqual(chosen['slot_id'],'g1:s0');self.assertTrue(meta['width_clamped'])
  chosen,meta=calibrated_opening(.09,slots);self.assertEqual(chosen['slot_id'],'g1:s1');self.assertTrue(meta['width_clamped'])
  self.assertIsNone(calibrated_opening(.05/1.7,slots))
 def test_source_pairing_and_both_flips_survive_narrow_old_common_width(self):
  raw,features=self.arrays();raw[0,12]=2.;original=features.copy()
  q,x,indices,receipt=make_hand_queries(raw,features,np.eye(4),self.policy,PROFILES['released-dh3-v1'],self.families,self.assets,{'fixture':'paired'})
  self.assertEqual(indices,[1]);self.assertEqual(len(q),8);self.assertEqual({v['if_flip']for v in q},{False,True})
  self.assertEqual({v['source_candidate_index']for v in q},{1});self.assertTrue(all(v['allowed_extra_depth_indices']==[0,1,2,3]for v in q))
  np.testing.assert_array_equal(features,original)
  for v,feature in zip(q,x):self.assertTrue(np.all(feature>=1000));self.assertEqual(v['canonical_CGR480_identity']['shape'],[480])
 def test_per_extra_insertion_and_approach_sign(self):
  raw,f=self.arrays();raw=raw[:1].copy();f=f[:1].copy();raw[:,14]=.055;raw[:,3:12]=np.array([[0,1,0],[0,0,-1],[-1,0,0]]).reshape(-1)
  q,_,_,_=make_hand_queries(raw,f,np.eye(4),self.policy,PROFILES['released-dh3-v1'],self.families,self.assets,{'fixture':'down'})
  self.assertEqual(len(q),8)
  for v in q:self.assertEqual(v['allowed_extra_depth_indices'],[0,1,2])
 def test_linear_approach_minimum_at_endpoint_and_exact_boundary(self):
  palm=np.eye(4);palm[2,3]=.1
  for a in [[0,0,1],[0,0,-1],[1,0,0]]:
   got=plane_minimum(self.vertices,palm,np.asarray(a))
   brute=min((self.vertices@palm[:3,:3].T+palm[:3,3]-t*np.asarray(a))[:,2].min()for t in np.linspace(0,.1,101))
   self.assertAlmostEqual(got,brute,places=15)
  p=np.eye(4);p[2,3]=.0005
  self.assertEqual(plane_minimum([[0,0,0]],p,np.array([1,0,0])),.0005)
  p[2,3]=np.nextafter(.0005,0);self.assertLess(plane_minimum([[0,0,0]],p,np.array([1,0,0])),.0005)
 def test_unsupported_extra_cannot_consume_global_budget_or_score_gate(self):
  q=[dict(query_index=0,raw_depth_index=0,allowed_extra_depth_indices=[0],all_head_outputs20=[.9,100.,0,0]+[0]*16),
     dict(query_index=1,raw_depth_index=0,allowed_extra_depth_indices=[0],all_head_outputs20=[.95,0,0,0]+[0]*16)]
  p=dict(PROFILES['released-dh3-v1'],score_topk=1,collision_topk=1)
  rows,short=expand_hand_queries(q,p);self.assertEqual(len(rows),2);self.assertEqual(short[0]['query_index'],1)
  for item in q:item['all_head_outputs20'][0]=float(np.float32(.85))
  self.assertEqual(expand_hand_queries(q,p)[1],[])
 def test_empty_domain_and_feature_mismatch_rejected(self):
  raw,f=self.arrays();raw[:,14]=-1
  q,x,i,r=make_hand_queries(raw,f,np.eye(4),self.policy,PROFILES['released-dh3-v1'],self.families,self.assets,{'fixture':'empty'})
  self.assertEqual((q,x,i),([],[],[]))
  with self.assertRaises(ValueError):make_hand_queries(raw,f[:-1],np.eye(4),self.policy,PROFILES['released-dh3-v1'],self.families,self.assets,{'fixture':'mismatch'})
if __name__=='__main__':unittest.main()
