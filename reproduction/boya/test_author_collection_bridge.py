"""Source-bound bridge checks; no simulator or physical grasp is launched."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import author_collection_bridge as bridge
from author_action_semantics import ActionContract
from author_collect import expanded_planned_hand, _candidate_id

ROOT = bridge.ROOT
BASE = ROOT/'outputs/boya_author_collector_20261114_v1'
CAL = ROOT/'outputs/boya_adapter_preparation_20261105_v2/calibration.json'
OBS = ROOT/'outputs/egad_external12_20261107_v1/observations/egad_A0/observation.json'


class AuthorCollectionBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared_path = BASE/'final_v4_g6/PREPARED.json'
        cls.candidate = json.loads(cls.prepared_path.read_text())['candidates'][0]
        cls.observation = json.loads(OBS.read_text())
        cls.contract = ActionContract(CAL)

    def pool(self, path):
        return bridge.bridge_candidate(self.candidate, path, calibration=CAL,
            object_root=self.observation['object_root'], world_T_object=self.observation['world_T_object'],
            prepared_path=self.prepared_path)

    def test_minimal_pool_preserves_fixed_choice_endpoints_and_depth_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'pool'; generation = self.pool(output)
            self.assertEqual(generation['selected_indices'], [0])
            self.assertFalse(generation['source_pose_relocated'])
            self.assertFalse(generation['mapping']['depth_added_again'])
            self.assertFalse(generation['additional_scene_collision_checked'])
            self.assertLess(generation['mapping']['max_matrix_roundoff'], 2e-6)
            with np.load(output/'candidates.npz', allow_pickle=False) as data:
                self.assertEqual(str(data['collection_candidate_sha256']), self.candidate['candidate_sha256'])
                np.testing.assert_array_equal(data['proposal_palm'][0, :3, 3], np.asarray(self.candidate['world_T_palm'])[:3, 3])
                np.testing.assert_array_equal(data['proposal_q'][0], self.candidate['action_contract']['opened_independent_rad'])
                np.testing.assert_array_equal(data['proposal_q_close'][0], self.candidate['action_contract']['closed_independent_rad'])
                np.testing.assert_array_equal(data['canonical_cgr480'][0], self.candidate['canonical_cgr480'])

    def test_changed_grasp_or_endpoint_cannot_be_rehashed_into_same_contract(self):
        for mutation in ('palm', 'contract'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                bad = copy.deepcopy(self.candidate)
                if mutation == 'palm': bad['world_T_palm'][0][3] += .01
                else: bad['action_contract']['closed_independent_rad'][0] += .1
                bad['candidate_sha256'] = _candidate_id(bad)
                with self.assertRaises(ValueError):
                    bridge.bridge_candidate(bad, Path(tmp)/'bad', calibration=CAL,
                        object_root=self.observation['object_root'], world_T_object=self.observation['world_T_object'])

    def test_real_native_24_column_plan_has_exact_source_20_hand_expansion(self):
        path = BASE/'bridge_g6_arm/native/commands.npz'
        with np.load(path, allow_pickle=False) as data: arrays = {k: data[k].copy() for k in data.files}
        self.assertEqual(len(arrays['joint_names']), 24)
        expanded, receipt = expanded_planned_hand(arrays, self.contract)
        self.assertEqual(len(expanded['joint_names']), 28)
        self.assertEqual(len(receipt['derived_mimic_columns']), 4)
        self.assertFalse(receipt['physical_joint_commands_observed'])
        names = list(arrays['joint_names'])
        motors = arrays['joint_target'][:, [names.index(n) for n in self.contract.names]]
        result = self.contract.validate_command_arrays(expanded, slot_id=self.candidate['slot_id'], motor_targets=motors,
            candidate_row15=self.candidate['actual_row15_camera'], world_T_camera=self.candidate['world_T_camera'],
            extra_depth_m=self.candidate['extra_depth_m'])
        self.assertTrue(result['all_hand_commands_exact'])
        self.assertFalse(result['physical_semantics_validated'])

    def test_fixed_choice_no_plan_does_not_try_other_top10_or_create_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = Path(tmp)/'pool'; self.pool(pool)
            repair = Path(tmp)/'primary_repair.json'; repair.write_text('{"source": "synthetic bridge argument check"}')
            cfg = json.loads((ROOT/'reproduction/boya/backend.collection.json').read_text())
            cfg['collision_refinement'] = str(repair)
            config = Path(tmp)/'backend.json'; config.write_text(json.dumps(cfg))
            def reject(command, **kwargs):
                native = Path(command[command.index('--output-dir')+1]); native.mkdir()
                (native/'admission_failure.json').write_text(json.dumps(dict(complete=True,
                    kind='arm_plan_rejected', physical_executed=False)))
                return SimpleNamespace(returncode=1)
            with patch.object(bridge.subprocess, 'run', side_effect=reject) as mocked:
                result = bridge.prepare_arm(pool, Path(tmp)/'planning', backend_config=config)
            self.assertEqual(mocked.call_count, 1)
            command = mocked.call_args.args[0]
            self.assertEqual(command[command.index('--index')+1], '0')
            self.assertEqual(command[command.index('--collision-refinement')+1], str(repair))
            self.assertIn(str(repair), result['input_sha256'])
            self.assertEqual(result['status'], 'no_plan')
            self.assertEqual(result['labels'], 0); self.assertEqual(result['physics_steps'], 0)


if __name__ == '__main__': unittest.main()
