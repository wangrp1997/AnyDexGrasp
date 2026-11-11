import copy
import json
import math
import unittest
import numpy as np
from shared_score_metrics import evaluate_scores, paired_object_bootstrap

def pair_metrics(mapping):
    labels, scores, objects, trials = [], [], [], []
    for name, pair in mapping.items():
        labels.extend([1, 0]);scores.extend(pair);objects.extend([name, name])
        trials.extend([name + '-positive', name + '-negative'])
    return evaluate_scores(labels, scores, objects, trials)

class SharedScoreMetricsTests(unittest.TestCase):
    def test_hand_computed_auc_ties_and_losses(self):
        r=evaluate_scores([0,1,0,1],[.1,.5,.5,.9],['x']*4,list('abcd'))
        self.assertEqual((r['count'],r['pos'],r['neg']),(4,2,2))
        self.assertAlmostEqual(r['global_auc'],.875)
        self.assertAlmostEqual(r['macro_object_auc'],.875)
        self.assertAlmostEqual(r['raw_mse'],.13)
        self.assertAlmostEqual(r['clipped_brier'],.13)
        self.assertAlmostEqual(r['clipped_logloss'],-(2*math.log(.9)+2*math.log(.5))/4)
        self.assertEqual(r['per_object'][0]['top1_selected_trial_id'],'d')

    def test_macro_denominators_exclude_single_class_auc_only(self):
        r=evaluate_scores([1,0,1,0,0,1,1],[.5,.5,.2,.8,.7,.1,.3],
                          ['a','a','b','b','b','c','d'],list('ABCDEFG'))
        self.assertEqual((r['object_count'],r['mixed_object_count'],r['single_class_object_count']),(4,2,2))
        self.assertEqual((r['positive_only_object_count'],r['negative_only_object_count']),(2,0))
        for key,value in [('macro_object_auc',.25),('global_auc',.5/12),
                          ('macro_top_tie_mean_label',.625),('macro_mixed_top_tie_mean_label',.25),
                          ('macro_object_brier',.535),('clipped_brier',.51)]:
            self.assertAlmostEqual(r[key],value)
        self.assertIsNone(r['per_object'][2]['pair_auc'])

    def test_top_tie_mean_is_not_trial_id_selection(self):
        for trials in (['z','a','m'],['a','z','m']):
            r=evaluate_scores([0,1,1],[.9,.9,.1],['x']*3,trials)
            self.assertEqual(r['macro_top_tie_mean_label'],.5)
            self.assertEqual(r['macro_mixed_top_tie_mean_label'],.5)
            self.assertEqual(r['per_object'][0]['top_tie_count'],2)
            self.assertEqual(r['per_object'][0]['top1_selected_trial_id'],'a')

    def test_clipping_for_losses_does_not_clip_rankings(self):
        r=evaluate_scores([0,1],[-2,3],['x','x'],['a','b'])
        self.assertEqual(r['raw_mse'],4);self.assertEqual(r['clipped_brier'],0)
        self.assertAlmostEqual(r['clipped_logloss'],-math.log1p(-1e-6))
        r=evaluate_scores([0,1],[2,3],['x','x'],['a','b'])
        self.assertEqual(r['global_auc'],1)
        self.assertEqual(r['per_object'][0]['top_tie_count'],1)

    def test_all_equal_scores_have_half_auc(self):
        r=evaluate_scores([1,0,0,1,0],[9]*5,['x']*5,list('abcde'))
        self.assertEqual(r['global_auc'],.5)
        self.assertEqual(r['macro_object_auc'],.5)
        self.assertEqual(r['macro_top_tie_mean_label'],.4)

    def test_single_class_and_empty_are_finite_json(self):
        for labels in ([0,0],[1,1]):
            r=evaluate_scores(labels,[.1,.9],['x','x'],['a','b'])
            self.assertIsNone(r['global_auc']);self.assertIsNone(r['macro_object_auc'])
            self.assertIsNone(r['macro_mixed_top_tie_mean_label'])
            self.assertEqual(r['mixed_object_count'],0);json.dumps(r,allow_nan=False)
        r=evaluate_scores([],[],[],[])
        self.assertEqual((r['count'],r['object_count']),(0,0))
        self.assertIsNone(r['raw_mse']);json.dumps(r,allow_nan=False)

    def test_input_arrays_are_not_mutated_and_order_does_not_change_ranking(self):
        y=np.array([0,1,0,1]);s=np.array([.1,.5,.5,.9]);yb=y.copy();sb=s.copy()
        a=evaluate_scores(y,s,[1]*4,list('abcd'));order=[3,1,0,2]
        b=evaluate_scores(y[order],s[order],[1]*4,np.array(list('abcd'))[order])
        np.testing.assert_array_equal(y,yb);np.testing.assert_array_equal(s,sb)
        for key in ('global_auc','macro_object_auc','macro_top_tie_mean_label'):
            self.assertEqual(a[key],b[key])
        self.assertEqual(a['per_object'][0]['object_id'],'1')

    def test_invalid_values_shapes_and_identity_are_rejected(self):
        invalid=[([0,1],[0,float('nan')],['x','x'],['a','b']),
          ([0,1],[0,float('inf')],['x','x'],['a','b']),
          ([0,float('nan')],[0,1],['x','x'],['a','b']),([0,.5],[0,1],['x','x'],['a','b']),
          ([[0],[1]],[0,1],['x','x'],['a','b']),([0,1],[[0],[1]],['x','x'],['a','b']),
          ([0,1],[0,1],[['x'],['x']],['a','b']),([0,1],[0],['x','x'],['a','b']),
          ([0,1],[0,1],['x','x'],['a','a']),([0,1],[0,1],['x',None],['a','b']),
          ([0,1],[0,1],['x','x'],['a','']),([0,1],[0,1e308],['x','x'],['a','b'])]
        for args in invalid:
            with self.subTest(args=args),self.assertRaises(ValueError):evaluate_scores(*args)

    def test_paired_bootstrap_common_mixed_objects_and_hand_delta(self):
        a=pair_metrics({'a':(1,0),'b':(0,1),'c':(.5,.5)})
        b=pair_metrics({'a':(.5,.5),'b':(1,0),'d':(1,0)})
        r=paired_object_bootstrap(a,b)
        self.assertEqual(r['object_ids'],['a','b'])
        self.assertEqual(r['common_mixed_object_count'],2)
        self.assertEqual(r['mean_delta_macro_pair_auc'],.25)
        self.assertEqual(r['ci95'],[-.5,1.])
        self.assertEqual((r['object_wins'],r['object_ties'],r['object_losses']),(1,0,1))
        self.assertEqual(r['excluded_reference_mixed_object_count'],1)
        self.assertEqual(r,paired_object_bootstrap(a,b));json.dumps(r,allow_nan=False)

    def test_bootstrap_tie_single_object_and_no_common_population(self):
        a=pair_metrics({'x':(.3,.3)});r=paired_object_bootstrap(a,a)
        self.assertEqual(r['ci95'],[0.,0.]);self.assertEqual(r['object_ties'],1)
        r=paired_object_bootstrap(a,pair_metrics({'y':(1,0)}))
        self.assertEqual(r['common_mixed_object_count'],0)
        self.assertIsNone(r['mean_delta_macro_pair_auc']);self.assertEqual(r['ci95'],[None,None])
        json.dumps(r,allow_nan=False)
        one=evaluate_scores([1],[.3],['x'],['one'])
        self.assertEqual(paired_object_bootstrap(one,a)['common_mixed_object_count'],0)

    def test_bootstrap_invalid_controls_and_metrics(self):
        a=pair_metrics({'x':(1,0)})
        for changed in ({'replicates':0},{'replicates':True},{'replicates':1.5},{'seed':-1},{'seed':True}):
            with self.assertRaises(ValueError):paired_object_bootstrap(a,a,**changed)
        bad=copy.deepcopy(a);bad['per_object'][0]['pair_auc']=float('nan')
        with self.assertRaises(ValueError):paired_object_bootstrap(a,bad)
        bad=copy.deepcopy(a);bad['per_object'].append(copy.deepcopy(bad['per_object'][0]))
        with self.assertRaises(ValueError):paired_object_bootstrap(a,bad)
        with self.assertRaises(ValueError):paired_object_bootstrap({},a)

if __name__=='__main__':unittest.main()
