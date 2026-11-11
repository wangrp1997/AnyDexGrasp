"""Selection safety contracts for bounded new-pose diagnosis."""
import unittest
from unittest.mock import patch
import numpy as np
import run_boya_candidate_diagnostic as m


def pose(x):
    return {k: np.array([x],dtype=float) for k in m.POSE_KEYS}


class SelectionTests(unittest.TestCase):
    def test_same_physical_pose_tolerates_reconstructed_roundoff(self):
        self.assertTrue(m.same_pose(pose(1.),pose(1.+1e-13)))
        self.assertFalse(m.same_pose(pose(1.),pose(1.+1e-5)))

    def test_old_pose_excluded_even_with_new_rank_and_different_file(self):
        data={'old':pose(1),'same':pose(1),'fresh':pose(2)}
        admitted=[dict(candidate_rank=0,candidate_dir='same'),dict(candidate_rank=1,candidate_dir='fresh')]
        with patch.object(m,'arrays',side_effect=lambda p:data[p]):
            selected,excluded=m.choose(admitted,[dict(candidate_dir='old',evaluation='old_evaluation')])
        self.assertEqual([c['candidate_rank'] for c in selected],[1])
        self.assertEqual(excluded[0]['historical_evaluations'],['old_evaluation'])

    def test_rank_order_fixed_budget_and_selected_pose_deduplication(self):
        data={str(i):pose(i) for i in range(5)}
        data['duplicate']=pose(1)
        admitted=[dict(candidate_rank=i,candidate_dir=str(i)) for i in reversed(range(5))]
        admitted.append(dict(candidate_rank=2,candidate_dir='duplicate'))
        with patch.object(m,'arrays',side_effect=lambda p:data[p]):
            selected,excluded=m.choose(admitted,[])
        self.assertEqual([c['candidate_rank'] for c in selected],[0,1,2])
        self.assertTrue(any(c['duplicate_selected'] for c in excluded))

    def test_budget_above_three_rejected(self):
        with self.assertRaises(ValueError):
            m.choose([],[],4)

    def test_short_pool_does_not_invent_replacement_candidates(self):
        with patch.object(m,'arrays',return_value=pose(1)):
            selected,_=m.choose([dict(candidate_rank=7,candidate_dir='one')],[])
        self.assertEqual(len(selected),1)


if __name__=='__main__':
    unittest.main()
