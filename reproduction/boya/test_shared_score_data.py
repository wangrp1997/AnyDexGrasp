"""Leakage, metadata whitelist and numeric contracts."""
import unittest
import numpy as np
from shared_score_data import metadata_features, grouped_folds, fit_standardizer, transform

def binding():
    return dict(multifinger_pose_finger_type=6,two_fingers_pose_depth_type=2,
                multifinger_pose_depth_type=3,if_flip=True,raw_width_m=.05,
                raw_depth_m=.02,extra_depth_m=.03,aperture={'actual_ref_width_m':.085})

class SharedDataContracts(unittest.TestCase):
    def test_same_mesh_all_gestures_and_collections_stay_together(self):
        mesh=np.repeat(['a','b','c','d','held'],3)
        split=np.repeat(['train']*4+['internal'],3)
        folds=grouped_folds(mesh,split,2,7);checks=[]
        for fold in folds:
            fit=fold['fit_indices'];check=fold['check_indices'];checks.extend(check)
            self.assertFalse(set(mesh[fit]) & set(mesh[check]))
            self.assertTrue(np.all(split[fit]=='train') and np.all(split[check]=='train'))
            for name in set(mesh[check]):self.assertEqual(sum(mesh[check]==name),3)
        self.assertEqual(sorted(checks),list(range(12)))

    def test_declared_empty_meshes_preserved_in_round_robin(self):
        folds=grouped_folds(['a','b','held'],['train','train','internal'],2,4,
                            declared_train_meshes=['a','b','empty1','empty2'])
        self.assertEqual([len(f['check_meshes']) for f in folds],[2,2])
        self.assertEqual(set().union(*(set(f['check_meshes']) for f in folds)),{'a','b','empty1','empty2'})

    def test_cross_split_mesh_is_rejected(self):
        with self.assertRaises(ValueError):
            grouped_folds(['a','a','b'],['train','internal','train'],2)

    def test_fit_only_statistics_and_unscaled_onehot(self):
        x=np.array([[1.,0.,10.],[3.,1.,10.],[1000.,1.,-999.]],dtype=np.float32)
        old=x.copy();s=fit_standardizer(x,np.array([0,1]),[0,2]);y=transform(x,s)
        np.testing.assert_array_equal(x,old)
        np.testing.assert_allclose(s['mean'],[2.,0.,10.]);np.testing.assert_allclose(s['scale'],[1.,1.,1.])
        np.testing.assert_array_equal(y[:,1],x[:,1]);self.assertEqual(float(y[2,0]),998.)
        self.assertEqual(y.dtype,np.dtype('float32'))

    def test_metadata_layout_and_whitelist(self):
        b=binding();x=metadata_features(b,6,11)
        self.assertEqual(x.shape,(16,));self.assertEqual(x.dtype,np.dtype('float32'))
        np.testing.assert_array_equal(np.flatnonzero(x[:13]),[1,6,12])
        np.testing.assert_allclose(x[13:],[1,.05,.085])
        b.update(object_id='secret',label=1,world_T_object=np.full((4,4),np.nan),post_execution_contact='secret')
        np.testing.assert_array_equal(x,metadata_features(b,6,11))

    def test_metadata_bad_values_and_binding_mismatch_rejected(self):
        for key,value in [('if_flip',1),('raw_width_m',float('nan')),('raw_width_m',0.),
                          ('raw_width_m',50.),('two_fingers_pose_depth_type',0),('extra_depth_m',.02)]:
            b=binding();b[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                metadata_features(b,6,11)
        for gesture,target in [(2,11),(6,20),(6,-1)]:
            with self.assertRaises(ValueError):metadata_features(binding(),gesture,target)

    def test_standardizer_rejects_bad_indices_and_nonfinite(self):
        x=np.zeros((3,2),dtype=np.float32)
        for indices in [[],[3],[1,1]]:
            with self.assertRaises(ValueError):fit_standardizer(x,np.asarray(indices,dtype=int))
        x[1,0]=np.nan
        with self.assertRaises(ValueError):fit_standardizer(x,np.array([0,2]))

if __name__=='__main__':unittest.main()
