import unittest
from unittest.mock import patch
import numpy as np
import author_mjx_policy_compare_worker as W


class RankingTests(unittest.TestCase):
    def test_head_ties_preserve_original_CGR_order(self):
        np.testing.assert_array_equal(W.ranking_order([.2,.8,.8,0]),[1,2,0,3])
        np.testing.assert_array_equal(W.ranking_order([0,0,0]),[0,1,2])

    def test_invalid_scores_cannot_silently_pick_a_candidate(self):
        for values in [[np.nan,0],[np.inf,1],[[.2,.3]]]:
            with self.assertRaises(ValueError):W.ranking_order(values)


if __name__=='__main__':unittest.main()
