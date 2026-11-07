"""Read-only, no-physics audit of the three frozen Boya maps and commands."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'reproduction/boya'))
from asset_adapter import BoyaAssets, cgr_reference_pose, map_slot
from run_boya_allegro_step_control import command_sequence, BASE_COMMAND_SEQUENCE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inputs = {}

    def freeze(path):
        path = Path(path).resolve()
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def read(path):
        return json.loads(freeze(path).read_text())

    def arrays(path):
        with np.load(freeze(path), allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    assets = BoyaAssets()
    base = ROOT / 'outputs/egad_floating_three_20261108_v1'
    calibration = read(ROOT / 'outputs/boya_adapter_preparation_20261105_v2/calibration.json')
    slots = {s['slot_id']: s for s in calibration['slots']}
    cases = []
    for oid in ['egad_A6', 'egad_A3', 'egad_E0']:
        job = read(base / 'jobs' / (oid + '.json'))
        pool = Path(job['candidate_dir'])
        g = read(pool / 'generation.json')
        saved = arrays(pool / 'candidates.npz')
        trace = arrays(base / 'execution' / oid / 'trace.npz')
        row = g['records'][0]
        obs = read(g['source_observation'])
        raw = arrays(Path(g['source_frame']) / 'raw_camera_candidates.npz')['candidates'][row['source_candidate_index']].copy()
        if row['if_flip']:
            raw[3:12].reshape(3, 3)[:, 1:3] *= -1
        slot = slots[row['source_slot']]
        palm = map_slot(raw, obs['T_WC'], slot, insertion_m=row['additional_depth_m'])
        ref = cgr_reference_pose(raw, obs['T_WC'], insertion_m=row['additional_depth_m'])
        approach = cgr_reference_pose(raw, obs['T_WC'])[:3, 2]
        np.testing.assert_allclose(palm, saved['proposal_palm'][0], atol=1e-12, rtol=0)
        np.testing.assert_allclose(approach, saved['proposal_approach_direction'][0], atol=1e-12, rtol=0)
        np.testing.assert_allclose(palm @ slot['palm_T_reference'], ref, atol=1e-12, rtol=0)
        family = assets.families[row['gesture_id']]
        opened, closed = saved['proposal_q'][0], saved['proposal_q_close'][0]
        np.testing.assert_array_equal(opened, slot['independent_q'])
        np.testing.assert_array_equal(closed, family.nominal)
        from canondex.grasp.template_gripper_closure import TwoSidedClosure
        closure = TwoSidedClosure.from_template(family, opened, closed)
        data = dict(palm=palm, direction=approach, opened=opened, closed=closed, closure=closure)
        old = BASE_COMMAND_SEQUENCE(data)
        staircase = command_sequence(data)
        motor_ix = [list(trace['joint_names']).index(n) for n in assets.hand.names]
        np.testing.assert_array_equal(old['motor_target'].astype(np.float32), trace['joint_target'][:, motor_ix])
        np.testing.assert_allclose(old['desired_palm'], trace['desired_palm'], atol=2e-14, rtol=0)
        close_ix = np.flatnonzero(trace['phase'] == 'close')
        start_q = trace['joint_pos'][close_ix[0] - 1, motor_ix]
        last_q = trace['joint_pos'][close_ix[-1], motor_ix]
        contacts = [assets.reference_contacts[k] for k in ('th', 'ff')]
        def pads(q):
            return assets.hand.evaluate(q, [c['collision_link'] for c in contacts],
                [c['point_link_m'] for c in contacts], [c['normal_link'] for c in contacts], jacobian=False)[0]
        p0, p1 = pads(opened), pads(closed)
        actual = read(base / 'execution' / oid / 'evaluation.json')
        caps = {n: v['native']['joint_effort_limit'] for n, v in actual['audit']['actuation']['joints'].items()}
        cases.append(dict(object_id=oid, gesture_id=row['gesture_id'], source_slot=row['source_slot'],
            raw_width_m=row['raw_width_m'], opening_width_m=slot['reference_width_m'],
            raw_depth_m=row['raw_depth_m'], extra_depth_m=row['additional_depth_m'],
            saved_palm_reconstructed=True, reference_frame_reconstructed=True,
            all_2100_original_independent_targets_bitwise_match=True,
            reference_closing_axis='thumb-to-index equals flipped CGR +Y by declared convention',
            original_author_common_axis_sign='center-to-thumb equals CGR +Y; two-finger 180 degree ambiguity; Boya labels use their own declared convention',
            max_measured_opening_joint_error_at_closure_rad=float(abs(start_q - opened).max()),
            max_actual_close_joint_error_rad=float(abs(last_q - closed).max()),
            commanded_pad_midpoint_shift_m=float(np.linalg.norm(p1.mean(0) - p0.mean(0))),
            commanded_open_width_m=float(np.linalg.norm(p0[1] - p0[0])),
            commanded_close_width_m=float(np.linalg.norm(p1[1] - p1[0])),
            native_effort_caps_nm=caps,
            source_staircase_only_changes_close_motor_targets=all(np.array_equal(old[k], staircase[k]) for k in old if k != 'motor_target'),
            changed_motor_rows=int(np.any(old['motor_target'].astype(np.float32) != staircase['motor_target'].astype(np.float32), axis=1).sum()),
            original_actual_success=actual['task_success']))
    sources = [Path(__file__), ROOT/'reproduction/boya/asset_adapter.py',
               ROOT/'reproduction/boya/run_boya_allegro_step_control.py',
               ROOT/'ur_toolbox/ur_toolbox/robot/Allegro/Allegro.py',
               ROOT/'ur_toolbox/ur_toolbox/robot/ur.py']
    for path in sources:
        freeze(path)
    for path, digest in inputs.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('Source changed during read-only audit: ' + path)
    out = dict(schema='anydex_boya_mapping_closure_source_audit_v1', cases=cases,
               physics_steps=0, hardware_imports=0, model_forwards=0, training_updates=0,
               measured_torque_trace_available=False,
               conclusions=['No demonstrated saved-frame/depth/flip/nominal-endpoint wiring error in these three cases.',
                            'Published Allegro closure is ten position steps, no active tactile or torque logic in close_gripper.',
                            'Position-path timing and hand-specific calibration are distinct from unidentified hardware equivalence.',
                            'Nominal pad midpoint drift is real but not by itself a bug; original lookup also uses opening geometry.',
                            'Staircase is isolated opt-in and has not been physically evaluated by this audit.'],
               input_and_source_sha256=inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    for c in cases:
        print(c['object_id'], 'opening_error_rad', c['max_measured_opening_joint_error_at_closure_rad'],
              'close_error_rad', c['max_actual_close_joint_error_rad'],
              'commanded_midpoint_drift_mm', c['commanded_pad_midpoint_shift_m'] * 1000,
              'staircase_changed_rows', c['changed_motor_rows'])


if __name__ == '__main__':
    main()
