"""Prevent source-row/feature drift when masking before top-K and scoring."""
from pathlib import Path
import copy
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,'/home/wangrenpeng/AnyDexGrasp/reproduction/boya')
from evaluate_s40_boya_port_v2 import make_port_queries
from deploy_score_first import make_queries, PROFILES

class DomainScoringContracts(unittest.TestCase):
    def setUp(self):
        self.raw=np.zeros((5,15),np.float32)
        self.raw[:,0]=[1.,.7,.99,.9,.6]
        self.raw[:,1]=[.2,.04,.2,.04,.04]
        self.raw[:,2]=[.0,.01,.02,.03,.04]
        self.raw[:,3:12]=np.eye(3).reshape(-1)
        self.raw[:,14]=.1
        self.features=np.zeros((5,2261),np.float32)
        self.features[:,-1]=self.raw[:,2]
        for i in range(5):self.features[i,240:720]=np.arange(480,dtype=np.float32)+i*1000
        self.config=dict(world_workspace_aabb_m=[[-1,-1,0],[1,1,1]],
            approach_world_cone=dict(axis_world=[1,0,0],cos_threshold=.9),raw_width_bounds_m=[.01,.1])
        self.profile=copy.deepcopy(PROFILES['released-dh3-v1']);self.profile['cgr_limit']=2

    def test_mask_before_topk_preserves_original_rows_features_and_depth(self):
        raw=self.raw.copy();features=self.features.copy()
        queries,inputs,indices,domain=make_port_queries(raw,features,np.eye(4),self.config,self.profile,{'source':'fixture'})
        self.assertEqual(domain.keep_indices.tolist(),[1,3,4])
        self.assertEqual(indices,[3,1])  # Top raw ranks 0/2 are outside domain.
        self.assertEqual(len(queries),16)
        for i in indices:
            for gesture in [1,6,7,8]:
                self.assertEqual({q['if_flip']for q in queries if q['source_candidate_index']==i and q['gesture_id']==gesture},{False,True})
        for q,x in zip(queries,inputs):
            i=q['source_candidate_index']
            expected_q,expected_x,_=make_queries(raw[i:i+1],features[i:i+1],self.profile,'both')
            reference=next(k for k in expected_q if k['gesture_id']==q['gesture_id'] and k['if_flip']==q['if_flip'])
            self.assertEqual(q['raw_depth_index'],reference['raw_depth_index'])
            self.assertEqual(q['if_flip'],reference['if_flip'])
            np.testing.assert_array_equal(x,expected_x[reference['query_index']])
        np.testing.assert_array_equal(raw,self.raw)
        np.testing.assert_array_equal(features,self.features)

    def test_empty_domain_has_no_queries(self):
        self.config['raw_width_bounds_m']=[.3,.4]
        q,x,i,d=make_port_queries(self.raw,self.features,np.eye(4),self.config,self.profile,{'source':'fixture'})
        self.assertEqual((q,x,i),([],[],[]));self.assertEqual(d.receipt['counts']['joint_pass'],0)

    def test_misaligned_or_nonfinite_features_fail_before_scoring(self):
        with self.assertRaises(ValueError):
            make_port_queries(self.raw,self.features[:-1],np.eye(4),self.config,self.profile,{'source':'fixture'})
        self.features[0,0]=np.nan
        with self.assertRaises(ValueError):
            make_port_queries(self.raw,self.features,np.eye(4),self.config,self.profile,{'source':'fixture'})

if __name__=='__main__':unittest.main()
