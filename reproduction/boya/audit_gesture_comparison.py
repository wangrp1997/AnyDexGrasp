#!/usr/bin/env python3
"""Independent saved-trace audit for the six-gesture development comparison.

Reads commands, recorded states, source object vertices, URDF and receipts only.
Never imports Isaac, a planner, a controller or a network; never steps physics.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def jsha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def joint_contract(path):
    result = {}
    for joint in ET.parse(path).getroot().findall('joint'):
        result[joint.get('name')] = dict(type=joint.get('type'),
            **{name: dict(joint.find(name).attrib) if joint.find(name) is not None else None
               for name in ['parent', 'child', 'origin', 'axis', 'limit', 'mimic']})
    return result


def audit_one(row, queue_row, object_result, vertices_cache):
    source_plan = Path(queue_row['plan_dir'])
    plan = load(source_plan/'plan.json')
    trial = Path(object_result['output_dir'])/'trials'/row['output_identity']
    execution = Path(row['execution_dir'])
    native = load(execution/'evaluation.json')
    bound = load(trial/'bound_evaluation.json')
    record = load(trial/'training_record.json')
    summary = load(trial/'RESULTS.json')
    binding = load(queue_row['binding_json'])
    source_commands = source_plan/'commands.npz'
    copied_commands = Path(record['execution_artifacts']['commands_npz'])
    command = arrays(source_commands)
    copied = arrays(copied_commands)
    trace = arrays(execution/'trace.npz')
    checks = {}
    def check(name, condition):
        checks[name] = bool(condition)
    names = list(map(str, trace['joint_names']))
    columns = [names.index(str(name)) for name in command['joint_names']]
    check('all_2100_physical_steps_and_clock',
        len(trace['time_s']) == len(trace['phase']) == native['total_physics_steps'] == row['physics_steps'] == record['total_physics_steps'] == 2100
        and native['complete'] is True and native['dt_s'] == .004
        and np.allclose(trace['time_s'], np.arange(1, 2101)*.004, atol=1e-14, rtol=0))
    check('every_command_array_and_staged_bytes_unchanged',
        sha(source_commands) == sha(copied_commands) == queue_row['commands_sha256'] == row['commands_sha256'] == native['commands_sha256']
        and command.keys() == copied.keys()
        and all(np.array_equal(command[key], copied[key]) and command[key].dtype == copied[key].dtype for key in command))
    check('recorded_actual_control_arrays_match_original',
        np.array_equal(trace['joint_target'][:, columns], command['joint_target'].astype(np.float32))
        and np.array_equal(trace['phase'], command['phase'])
        and np.array_equal(trace['desired_palm'], command['desired_palm']))
    # The remaining four recorded coordinates are the source passive mimics.
    source_contract = joint_contract(plan['arguments']['robot_urdf'])
    local_contract = joint_contract(execution/'ur16e_boya.urdf')
    structural_match = source_contract == local_contract
    check('executed_URDF_joint_structure_matches_source', structural_match)
    mimic_specs = {name: info['mimic'] for name, info in local_contract.items() if info['mimic'] is not None}
    passive_targets_match = all(name in names and spec['joint'] in names and
        np.array_equal(trace['joint_target'][:, names.index(name)],
            trace['joint_target'][:, names.index(spec['joint'])]*float(spec.get('multiplier', '1'))+float(spec.get('offset', '0')))
        for name, spec in mimic_specs.items())
    check('all_28_actual_targets_including_four_source_mimics',
        trace['joint_target'].shape == (2100, 28) and len(command['joint_names']) == 24
        and len(mimic_specs) == 4 and passive_targets_match)
    digest = jsha({key: value for key, value in binding.items() if key != 'trial_binding_sha256'})
    check('canonical_binding_type_depth_and_initial_plan_identity',
        digest == binding['trial_binding_sha256'] == row['trial_binding_sha256'] == plan['trial_binding_sha256'] == str(command['trial_binding_sha256'].item())
        and binding == record['binding'] == load(trial/'trial_binding.json')
        and row['gesture_id'] == binding['multifinger_pose_finger_type'] == native['gesture_id']
        and row['extra_depth_index'] == binding['multifinger_pose_depth_type'] == 0
        and binding['extra_depth_m'] == 0.
        and np.array_equal(plan['source_palm'], binding['execution_spec']['world_T_palm'])
        and np.array_equal(plan['world_T_object'], binding['execution_spec']['world_T_object']))
    check('native_trace_evaluation_and_record_artifact_SHA',
        sha(execution/'trace.npz') == native['trace_sha256']
        and sha(execution/'evaluation.json') == bound['original_native_evaluation_sha256']
        and sha(trial/'bound_evaluation.json') == record['execution_artifacts']['evaluation_sha256']
        and sha(copied_commands) == record['execution_artifacts']['commands_sha256']
        and all(bound[key] == value for key, value in native.items() if key != 'success_definition'))
    check('saved_original_eleven_execution_guards_and_eligible_label',
        len(row['guards']) == 11 and all(row['guards'].values())
        and row['guards'] == summary['guards'] == bound['queue_binding_guards']
        and row['label_eligible'] is True and record['label_eligible'] is True
        and record['result'] in [0, 1])
    check('all_numeric_trace_arrays_finite',
        all(np.isfinite(value).all() for value in trace.values() if value.dtype.kind in 'fiu'))
    object_root = Path(native['object_root'])
    mesh = object_root/'mesh/simplified.obj'
    key = (str(mesh), float(native['object_scale']))
    if key not in vertices_cache:
        vertices_cache[key] = np.array([[float(x) for x in line.split()[1:4]]
            for line in mesh.read_text().splitlines() if line.startswith('v ')])*native['object_scale']
    vertices = vertices_cache[key]
    source_pose_path = next((p for p in queue_row.get('input_sha256', {}) if p == str(mesh)), None)
    check('object_source_mesh_and_scale_bound_to_prepared_plan',
        source_pose_path is not None and sha(mesh) == queue_row['input_sha256'][source_pose_path]
        and plan['arguments']['object_root'] == native['object_root']
        and Path(native['object_root']).name == row['object_id'])
    phases = trace['phase']; holds = np.flatnonzero(phases == 'hold')
    if not len(holds):
        raise ValueError('No original last-hold reference state: ' + row['trial_id'])
    poses = trace['object_pose_xyzw'].astype(float)
    reference_z = float(poses[holds[-1], 2])
    rise = float(poses[-1, 2]-reference_z)
    bottom = float(np.min(vertices @ Rotation.from_quat(poses[-1, 3:]).as_matrix()[2] + poses[-1, 2]))
    loads = trace['finger_loads_n'][-1]
    thumb = float(loads[0]); opponent = float(loads[1:].sum())
    bearing = bool(thumb > .001 and opponent > .001)
    held = bool(phases[-1] == 'lift' and rise >= .005 and bottom >= .005 and bearing)
    gates = native['gates']
    check('all_original_native_seven_gates_retained',
        set(gates) == {'joint_limits', 'mimics', 'penetration', 'palm_tracking', 'no_arm_object_support', 'fixed_base', 'complete_execution'}
        and gates == bound['gates'])
    success = bool(all(gates.values()) and held and rise >= .05)
    check('independent_source_mesh_bottom_last_hold_rise_loads_and_label',
        abs(rise-native['pickup']['final_reference_rise_m']) < 1e-12
        and abs(bottom-native['pickup']['final_bottom_clearance_m']) < 1e-10
        and held == native['pickup']['final_state_still_lifted_and_bearing']
        and success == native['task_success'] == bool(record['result'])
        and record['result'] == row['label'] == summary['label'])
    if not structural_match:
        raise ValueError('Joint structure mismatch: refusing independent joint-gate reconstruction')
    limit = 0.
    for column, name in enumerate(names):
        limits = local_contract[name]['limit']
        if limits and 'lower' in limits and 'upper' in limits:
            limit = max(limit, float(np.maximum(np.maximum(float(limits['lower'])-trace['joint_pos'][:, column],
                trace['joint_pos'][:, column]-float(limits['upper'])), 0.).max()))
    mimic = max(float(np.abs(trace['joint_pos'][:, names.index(name)]
        - float(spec.get('multiplier', '1'))*trace['joint_pos'][:, names.index(spec['joint'])]
        - float(spec.get('offset', '0'))).max()) for name, spec in mimic_specs.items())
    penetration = float(max(trace['object_depth_m'].max(), trace['robot_depth_m'].max()))
    check('trace_recomputed_three_native_physical_gates',
        (limit <= .002) == gates['joint_limits'] and (mimic <= .002) == gates['mimics'] and (penetration <= .001) == gates['penetration']
        and abs(limit-native['joint_limit_violation_rad']) < 1e-7
        and abs(mimic-native['mimic_error_rad']) < 1e-7
        and abs(penetration-native['penetration_m']) < 1e-12)
    check('no_recorded_runtime_state_write_or_external_palm_wrench',
        native['state_writes_during_execution'] == 0 and native['external_palm_wrench_used'] is False)
    reasons = [name for name, passed in gates.items() if not passed]
    if rise < .05: reasons.append('final_reference_rise_below_50mm')
    if bottom < .005: reasons.append('final_mesh_bottom_below_5mm')
    if thumb <= .001: reasons.append('no_final_thumb_load_over_1mN')
    if opponent <= .001: reasons.append('no_final_opponent_sum_over_1mN')
    return dict(trial_id=row['trial_id'], object_id=row['object_id'], gesture_id=row['gesture_id'],
        label=record['result'], all_checks_passed=all(checks.values()), checks=checks,
        actual_physics_steps=2100, reference_rise_mm=rise*1000, source_mesh_bottom_mm=bottom*1000,
        final_thumb_load_n=thumb, final_opponent_sum_n=opponent, final_held=held, task_success=success,
        native_gate_failures=[name for name, passed in gates.items() if not passed], failure_reasons=reasons,
        joint_limit_violation_rad=limit, mimic_error_rad=mimic, penetration_m=penetration,
        source_mesh_sha256=sha(mesh), trace_sha256=sha(execution/'trace.npz'),
        commands_sha256=sha(source_commands), training_record_sha256=sha(trial/'training_record.json'))


def audit(args):
    start = time.monotonic(); root = args.run_dir.resolve(); repo = Path(__file__).resolve().parents[2]
    final_path = root/'RESULTS.json'
    if not final_path.is_file():
        raise ValueError('Final RESULTS.json is not ready; this audit does not poll or execute trials')
    final = load(final_path); prepared_path = root/'PREPARED.json'; prepared = load(prepared_path)
    if final.get('status') != 'complete':
        raise ValueError('Require a terminal complete comparison before final audit')
    protected = {str(p): sha(p) for p in [final_path, prepared_path, root/'training_records.jsonl']}
    if final['prepared_sha256'] != sha(prepared_path):
        raise ValueError('Comparison PREPARED SHA changed')
    queue = load(prepared['queue_manifest'])
    if sha(prepared['queue_manifest']) != prepared['queue_manifest_sha256']:
        raise ValueError('Comparison queue SHA changed')
    queue_rows = {row['trial_id']: row for row in queue['trials']}
    rows = []; vertices_cache = {}; seen = set(); physical_steps = 0; clocks = 0
    for item in final['object_results']:
        sub = item['result']
        if item['status'] == 'no_ready_plans':
            if sub['trials']:
                raise ValueError('Zero-ready object unexpectedly has actual trials')
            continue
        if item['status'] != 'complete' or item['returncode'] != 0:
            raise ValueError('Object queue did not complete normally: ' + item['object_id'])
        for cohort_row in sub['cohorts']:
            cohort_path = Path(cohort_row['receipt_path']).parent if 'receipt_path' in cohort_row else Path(cohort_row['output_dir']) if 'output_dir' in cohort_row else None
            if cohort_path is None:
                # Executor stores the worker receipt itself at this fixed cohort path.
                cohort_path = Path(item['output_dir'])/'cohorts'/('cohort_'+str(cohort_row['cohort_index']).zfill(4)) if 'cohort_index' in cohort_row else None
            if cohort_path is None:
                raise ValueError('Unknown cohort receipt schema')
            cohort = load(cohort_path/'result.json')
            if not (cohort['complete'] and cohort['mode'] == 'serial' and cohort['device'] == 'cpu'
                and cohort_row['returncode'] == 0 and cohort['inputs_unchanged'] and cohort['sources_unchanged']):
                raise ValueError('Original resident worker receipt not complete')
            for clock in cohort['clocks']:
                assert clock['physics_ticks'] == clock['recorded_environment_steps'] == clock['integrated_environment_steps'] == clock['command_steps'] == 2100
                assert clock['all_streams_complete'] is True
                clocks += 1
            assert cohort['physics_ticks'] == cohort['environment_steps'] == len(cohort['clocks'])*2100
            physical_steps += cohort['physics_ticks']
        for row in sub['trials']:
            if row['trial_id'] in seen: raise ValueError('Repeated actual trial ID')
            seen.add(row['trial_id'])
            rows.append(audit_one(row, queue_rows[row['trial_id']], item, vertices_cache))
    checks = dict(all_prepared69_actual_trials_present=len(rows) == len(queue_rows) == prepared['ready_plans'] == 69,
        all69_full_control_and_physical_evidence_passed=all(row['all_checks_passed'] for row in rows),
        exactly69_original2100step_clocks=clocks == 69 and physical_steps == 69*2100 == final['actual_physics_steps'],
        all48_cells_and96_slots_retained=len(final['cells']) == 48 and sum(len(c['rounds']) for c in final['cells']) == 96,
        all27_unexecuted_slots_still_null=prepared['null_rounds'] == 27 and len(queue['nulls']) == 27)
    exported = [json.loads(line) for line in (root/'training_records.jsonl').read_text().splitlines()]
    exported_by_id = {row['collection_context']['trial_id']: row for row in exported}
    for item in final['object_results']:
        for row in item['result']['trials']:
            source = load(Path(item['output_dir'])/'trials'/row['output_identity']/'training_record.json')
            actual = dict(exported_by_id[row['trial_id']]); context = actual.pop('collection_context')
            if actual != source or context['gesture_id'] != row['gesture_id'] or context['object_id'] != row['object_id']:
                raise ValueError('Export changed an actual executed training record')
    positive = sum(row['label'] == 1 for row in rows); negative = sum(row['label'] == 0 for row in rows)
    checks['all_preparation_nulls_have_no_physical_label'] = all(row.get('label') is None and row.get('training_label') is None and row.get('actual_physics_steps', 0) == 0 for row in queue['nulls'])
    checks['final_labels_and_export_match_independent_reconstruction'] = len(exported) == len(exported_by_id) == final['label_count'] == 69 and positive == final['positive_labels'] and negative == final['negative_labels']
    # Recheck only original final summary/export, not the old 751-file source audit.
    old = repo/'outputs/boya_pilot8_20261105_v2'
    archive = repo/'reproduction/boya/evidence/pilot8_continuation_20261105'
    old_summary = load(old/'FINAL_SUMMARY.json')
    checks['old36_summary_and_JSONL_bytes_unchanged'] = sha(old/'FINAL_SUMMARY.json') == sha(archive/'FINAL_SUMMARY.json') and sha(old/'training_records.jsonl') == old_summary['training_records']['sha256'] and old_summary['aggregate']['actual_attempts'] == 36 and old_summary['aggregate']['positive_labels'] == 29 and old_summary['aggregate']['negative_labels'] == 7
    checks['protected_final_artifacts_unchanged_during_audit'] = all(sha(path) == digest for path, digest in protected.items())
    per_gesture = []
    for gesture in [2, 3, 5, 7, 9, 10]:
        group = [row for row in rows if row['gesture_id'] == gesture]
        per_gesture.append(dict(gesture_id=gesture, actual=len(group), positive=sum(row['label'] == 1 for row in group), negative=sum(row['label'] == 0 for row in group),
            reference_rise_mm_range=[min(row['reference_rise_mm'] for row in group), max(row['reference_rise_mm'] for row in group)],
            mesh_bottom_mm_range=[min(row['source_mesh_bottom_mm'] for row in group), max(row['source_mesh_bottom_mm'] for row in group)]))
    return dict(schema='anydex_boya_gesture_comparison_independent_physics_audit_v1',
        all_checks_passed=all(checks.values()), checks=checks,
        aggregate=dict(new_actual=69, positive=positive, negative=negative, physics_steps=physical_steps, zero_step_null_slots=27,
            old_actual_unchanged=36, old_positive_unchanged=29, old_negative_unchanged=7),
        per_gesture=per_gesture, trials=rows, final_artifacts_sha256=protected,
        old_summary_sha256=sha(old/'FINAL_SUMMARY.json'), old_JSONL_sha256=sha(old/'training_records.jsonl'),
        audit_script_sha256=sha(__file__), cost=dict(new_physics_steps=0, new_network_forwards=0, new_training_steps=0, elapsed_wall_s=time.monotonic()-start),
        boundary=['Eight development objects, not the formal 40+10 collection or an unseen-object estimate.',
            'Recorded complete original commands and all 2100 states are checked independently; source mesh is used only for offline label verification.',
            'Limits/mimics/penetration are reconstructed from matching executed/source URDF and saved states/depth channels.',
            'Other four original native gates are bound to their unchanged native evidence; detailed every-step illegal-contact pairs are not available in trace, so their contact measurement is not regenerated.',
            'No new simulation, observation, network, planning, controller or source modification.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[2]
    parser.add_argument('--run-dir', type=Path, default=repo/'outputs/boya_gesture_comparison_20261105_v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(all_checks_passed=result['all_checks_passed'], checks=result['checks'], aggregate=result['aggregate'], per_gesture=result['per_gesture'], output=str(args.output)), indent=2))
    raise SystemExit(0 if result['all_checks_passed'] else 1)


if __name__ == '__main__':
    main()
