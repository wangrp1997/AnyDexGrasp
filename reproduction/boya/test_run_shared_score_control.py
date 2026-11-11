import copy
import unittest
from run_shared_score_control import gate_result, select_candidate


def metric(auc, brier=.15, top=.8):
    one = {"macro_object_auc": auc, "macro_object_brier": brier,
           "macro_mixed_top_tie_mean_label": top}
    return {"object": dict(one), "same_observation": dict(one)}


class SharedControlTests(unittest.TestCase):
    def fixtures(self):
        cv = {"shared_linear": metric(.70), "shared_mlp64": metric(.75),
              "condition_linear": metric(.55), "object_prior": metric(.5, .25, .65)}
        internal = {"shared_mlp64": metric(.75), "original_mse": metric(.60),
                    "preactivation_mse": metric(.65), "condition_linear": metric(.55),
                    "object_prior": metric(.5, .25, .65)}
        folds = [{"metrics": copy.deepcopy(cv)} for _ in range(4)]
        return cv, internal, folds

    def test_selection_is_cv_only_and_tie_favors_linear(self):
        cv, _, _ = self.fixtures()
        self.assertEqual(select_candidate(cv), "shared_mlp64")
        cv["shared_linear"] = metric(.75)
        self.assertEqual(select_candidate(cv), "shared_linear")

    def test_all_predeclared_conditions_are_needed(self):
        cv, internal, folds = self.fixtures()
        self.assertTrue(gate_result("shared_mlp64", cv, internal, folds,
                                    {"gain_ci95": [.02, .3]})["passed"])
        self.assertFalse(gate_result("shared_mlp64", cv, internal, folds,
                                     {"gain_ci95": [-.02, .3]})["passed"])
        internal["shared_mlp64"]["object"]["macro_mixed_top_tie_mean_label"] = .7
        decision = gate_result("shared_mlp64", cv, internal, folds, {"gain_ci95": [.02, .3]})
        self.assertFalse(decision["passed"])
        self.assertEqual(decision["decision"], "stop_no_physical_promotion")

    def test_good_pooled_score_does_not_hide_fold_failures(self):
        cv, internal, folds = self.fixtures()
        for fold in folds[:2]:
            fold["metrics"]["shared_mlp64"] = metric(.50)
        result = gate_result("shared_mlp64", cv, internal, folds, {"gain_ci95": [.02, .3]})
        self.assertFalse(result["checks"]["cv_positive_gain_in_3_of_4_folds"])


if __name__ == "__main__":
    unittest.main()
