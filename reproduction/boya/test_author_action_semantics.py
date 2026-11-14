"""Contract mutation checks; no simulator, optimizer or source-controller import."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from author_action_semantics import ActionContract, CGR_R_REFERENCE, source_flip


class ActionSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = ActionContract()

    def arrays(self, slot_id='g6:distal_010:ff'):
        progress = np.r_[0., 0., .25, .5, .75, 1., 1.]
        closed = self.contract.closure_targets(slot_id, progress)
        return dict(joint_names=np.array(closed['joint_names']),
                    joint_target=closed['joint_target'].astype(np.float32),
                    phase=np.array(['approach', 'approach', 'close', 'close', 'close', 'close', 'hold']))

    def test_source_collection_flip_reverses_approach_but_deployment_retains_it(self):
        rows = np.zeros((2, 17)); rows[:, 4:13] = np.eye(3).ravel()
        rows[1, 4:13] = np.diag([-1., -1., 1.]).ravel()
        collection, flags_c = source_flip(rows, 'dh3_collection')
        deployment, flags_d = source_flip(rows, 'dh3_deployment')
        np.testing.assert_array_equal(flags_c, [True, False])
        np.testing.assert_array_equal(flags_d, flags_c)
        np.testing.assert_array_equal(collection[0, 4:13].reshape(3, 3)[:, 0], [-1., 0., 0.])
        np.testing.assert_array_equal(deployment[0, 4:13].reshape(3, 3)[:, 0], [1., 0., 0.])
        np.testing.assert_array_equal(collection[1], rows[1])
        np.testing.assert_array_equal(rows[0, 4:13], np.eye(3).ravel())

    def test_all_types_keep_inactive_and_nonflexion_commands_and_four_mimics(self):
        for gesture in (1, 6, 7, 8):
            with self.subTest(gesture=gesture):
                slot = f'g{gesture}:distal_000:ff'
                bound = self.contract.for_slot(slot)
                targets = self.contract.closure_targets(slot, [0., 1.])
                self.assertEqual(len(targets['joint_names']), 20)
                self.assertTrue(bound['numeric_contract_validated'])
                self.assertFalse(bound['physical_semantics_validated'])
                self.assertNotIn('palm', bound['active_groups'])
                for i, name in enumerate(targets['joint_names']):
                    if name[:2].lower() in bound['inactive_groups'] or name.endswith('J4') or name in ('THJ3', 'THJ4'):
                        self.assertEqual(targets['joint_target'][0, i], targets['joint_target'][1, i])
                for name, (driver, coefficient, offset) in self.contract.mimics.items():
                    np.testing.assert_array_equal(targets['joint_target'][:, targets['joint_names'].index(name)],
                        coefficient*targets['joint_target'][:, targets['joint_names'].index(driver)]+offset)

    def test_mutated_endpoint_or_inactive_coordinate_rejected(self):
        for field, index in [('close_q', 0), ('independent_q', 0), ('independent_q', 10)]:
            with self.subTest(field=field, index=index):
                calibration = copy.deepcopy(self.contract.calibration)
                slot = next(s for s in calibration['slots'] if s['slot_id'] == 'g6:distal_010:ff')
                slot[field][index] += .001
                with self.assertRaises(ValueError): ActionContract(calibration)

    def test_actual_commands_and_mimic_mutation(self):
        arrays = self.arrays()
        report = self.contract.validate_command_arrays(arrays, slot_id='g6:distal_010:ff')
        self.assertTrue(report['all_hand_commands_exact'])
        self.assertFalse(report['physical_semantics_validated'])
        for joint in ('THJ1', 'FFJ1', 'LFJ3'):
            with self.subTest(joint=joint):
                changed = copy.deepcopy(arrays)
                changed['joint_target'][3, list(changed['joint_names']).index(joint)] += .001
                with self.assertRaises(ValueError):
                    self.contract.validate_command_arrays(changed, slot_id='g6:distal_010:ff')

    def test_reference_midpoint_mutation_and_explicit_path_bypass_rejected(self):
        calibration = copy.deepcopy(self.contract.calibration)
        slot = next(s for s in calibration['slots'] if s['slot_id'] == 'g6:distal_010:ff')
        slot['palm_T_reference'][0][3] += .001
        with self.assertRaises(ValueError): ActionContract(calibration)
        arrays = self.arrays()
        motors = arrays['joint_target'][:, :16].astype(float)
        # Use exact expected float64 targets, then change an inactive finger in
        # both the expected and actual arrays: equality alone must not admit it.
        motors = self.contract.closure_targets('g6:distal_010:ff', [0., 0., .25, .5, .75, 1., 1.])['independent_target']
        motors[3, 10] += .001
        arrays['joint_target'][3, 10] = motors[3, 10]
        with self.assertRaises(ValueError):
            self.contract.validate_command_arrays(arrays, slot_id='g6:distal_010:ff', motor_targets=motors)

    def test_missing_duplicate_joint_or_unknown_phase_rejected(self):
        original = self.arrays()
        for kind in ('missing', 'duplicate', 'phase'):
            with self.subTest(kind=kind):
                arrays = copy.deepcopy(original)
                if kind == 'missing':
                    arrays['joint_names'] = arrays['joint_names'][:-1]
                    arrays['joint_target'] = arrays['joint_target'][:, :-1]
                elif kind == 'duplicate': arrays['joint_names'][-1] = arrays['joint_names'][0]
                else: arrays['phase'] = np.array(['unknown']*len(arrays['phase']))
                with self.assertRaises(ValueError): self.contract.validate_command_arrays(arrays, slot_id='g6:distal_010:ff')

    def test_approach_reference_depth_and_actual_row_binding(self):
        slot = 'g6:distal_010:ff'
        row = np.r_[.7, .06, .02, np.eye(3).ravel(), [.1, .2, .3]]
        palm = self.contract.mapped_palm(slot, row, np.eye(4), extra_depth_m=.03)
        reference = palm@np.asarray(self.contract.slots[slot]['palm_T_reference'])
        np.testing.assert_allclose(reference[:3, :3], CGR_R_REFERENCE, atol=1e-14)
        np.testing.assert_allclose(reference[:3, 3], [.15, .2, .3], atol=1e-14)
        arrays = self.arrays(slot)
        arrays['desired_palm'] = np.repeat(palm[None], len(arrays['phase']), axis=0)
        self.contract.validate_command_arrays(arrays, slot_id=slot, candidate_row15=row,
                                             world_T_camera=np.eye(4), extra_depth_m=.03)
        wrong = row.copy(); wrong[2] += .01
        with self.assertRaises(ValueError):
            self.contract.validate_command_arrays(arrays, slot_id=slot, candidate_row15=wrong,
                                                 world_T_camera=np.eye(4), extra_depth_m=.03)
        contract = self.contract.for_slot(slot)
        self.assertEqual(json.loads(json.dumps(contract)), contract)


if __name__ == '__main__':
    unittest.main()
