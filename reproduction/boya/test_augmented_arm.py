"""Reject accidental cohort/policy changes before expensive native initialization."""
import copy
import tempfile
from pathlib import Path
import unittest

from prepare_augmented_arm import CachedInputs
from prepare_augmented_arm_runtime import validate_fast6_protocol, OBJECT_IDS
from run_augmented_arm_external6 import read, OLD


class AugmentedArmContract(unittest.TestCase):
    def protocol(self):
        p=copy.deepcopy(read(OLD/'protocol.json'))
        p.update(schema='anydex_boya_augmented_arm6_protocol_v1',
            objects=[dict(object_id=o) for o in OBJECT_IDS],
            head_recipe='paired_frozen_original_and_preactivation')
        return p

    def test_exact_new_cohort_required(self):
        p=self.protocol()
        self.assertEqual(validate_fast6_protocol(p),list(OBJECT_IDS))
        p['objects'][0]['object_id']='egad_A1'
        with self.assertRaises(ValueError):validate_fast6_protocol(p)

    def test_attempt_budget_and_recipe_scope_required(self):
        for field,value in [('physical_attempts_per_object',2),('arm_budget_per_object',5),
                ('head_recipe','preactivation_selected_mse'),('training_writeback',True),
                ('outcome_dependent_retry',True),('object_mass_kg',.2),('object_friction',1.)]:
            with self.subTest(field=field):
                p=self.protocol();p[field]=value
                with self.assertRaises(ValueError):validate_fast6_protocol(p)

    def test_cached_hash_conflict_and_final_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'input';p.write_text('original')
            inputs=CachedInputs();h=inputs.freeze(p)
            self.assertEqual(inputs.freeze(p,h),h)
            with self.assertRaises(ValueError):inputs.freeze(p,'0'*64)
            p.write_text('mutated')
            with self.assertRaises(ValueError):inputs.verify()


if __name__=='__main__':unittest.main()
