"""Certify an existing native task stop without executing or relabelling it.

This adapter accepts only a complete, error-free native closeout whose saved
trace is an exact prefix of its original commands and whose first native stop
predicate is independently reproduced. A certificate is a terminal null, never
an unsuccessful (0) training example. Unknown or ambiguous evidence raises.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from execute_collection_queue_v2 import command_contract, trial_record_module

KNOWN_TASK_STOPS = frozenset(('object_pushed_before_closure', 'deep_collision_stop'))
_HASH_CACHE = {}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if key not in _HASH_CACHE:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                digest.update(block)
        _HASH_CACHE[key] = digest.hexdigest()
    return _HASH_CACHE[key]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(mapping):
    require(isinstance(mapping, dict) and bool(mapping), 'Missing frozen hash mapping')
    for path, expected in mapping.items():
        require(Path(path).is_file() and sha(path) == expected, 'Changed or absent frozen input/source: ' + path)


def certify_known_task_stop(row):
    """Return a source-bound terminal-null certificate; otherwise raise ValueError.

    ``row`` is an unchanged execute_collection_queue_v2.finish_trial result.
    Worker/source/input guards and original worker receipts are required. No
    caller-provided cohort/context is trusted or required.
    """
    try:
        return _certify(row)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError('Missing or invalid task-stop evidence: ' + str(error)) from error


def _certify(row):
    require(row.get('label') is None and row.get('label_eligible') is False,
            'Only ineligible null outcomes can be certified')
    required = ('native_scene_backend_confirmed', 'source_commands_unchanged',
                'native_echo_commands', 'worker_completed_without_error', 'inputs_and_sources_unchanged')
    require(all(row.get('guards', {}).get(key) is True for key in required),
            'Worker/backend/source/input guards not all confirmed')
    execution = Path(row['execution_dir']).resolve()
    cohort_dir = execution.parent
    executor_dir = cohort_dir.parent.parent
    ev_path, trace_path = execution/'evaluation.json', execution/'trace.npz'
    evaluation = read(ev_path)
    reason = evaluation.get('failure_reason')
    require(reason in KNOWN_TASK_STOPS, 'Unknown native stop reason: ' + str(reason))
    require(evaluation.get('complete') is True and evaluation.get('task_success') is False
            and not evaluation.get('error') and not evaluation.get('infrastructure_error'),
            'Native evaluation did not close normally')
    require(evaluation.get('gates', {}).get('complete_execution') is False,
            'This certificate is only for an incomplete original command sequence')
    require(evaluation.get('state_writes_during_execution') == 0,
            'State writes occurred during physical execution')
    require(not evaluation.get('tactile_feedback'), 'Feedback stops are outside this certificate contract')
    verify(evaluation['source_sha256'])
    require(sha(trace_path) == evaluation['trace_sha256'], 'Native trace SHA mismatch')
    receipt_path = cohort_dir/'result.json'
    receipt = read(receipt_path)
    require(receipt.get('complete') is True and not receipt.get('error')
            and receipt.get('inputs_unchanged') is True and receipt.get('sources_unchanged') is True,
            'Native worker closeout/source/input evidence invalid')
    prepared_path, result_path = executor_dir/'PREPARED.json', executor_dir/'RESULTS.json'
    prepared, result = read(prepared_path), read(result_path)
    verify(prepared['source_sha256']); verify(prepared['input_sha256'])
    require(result.get('source_sha256_unchanged') is True and result.get('input_sha256_unchanged') is True
            and not result.get('error') and not result.get('infrastructure_error'),
            'Executor source/input/error status invalid')
    summaries = [value for value in result.get('cohorts', [])
                 if Path(value['receipt_path']).resolve() == receipt_path]
    require(len(summaries) == 1 and summaries[0].get('returncode') == 0,
            'No unambiguous successful worker process return')
    saved = [value for value in result.get('trials', []) if value.get('trial_id') == row['trial_id']]
    require(len(saved) == 1 and saved[0] == row, 'Row differs from its original executor result')
    declared = [value for value in prepared['trials'] if value['trial_id'] == row['trial_id']]
    require(len(declared) == 1, 'Trial missing or repeated in prepared queue')
    declared = declared[0]
    plan_dir = Path(row['source_plan_dir']).resolve()
    require(Path(declared['plan_dir']).resolve() == plan_dir
            and Path(declared['binding_json']).resolve() == Path(row['binding_json']).resolve(),
            'Row plan/binding differs from prepared source')
    plan_path, command_path = plan_dir/'plan.json', plan_dir/'commands.npz'
    plan, binding = read(plan_path), read(row['binding_json'])
    trial_record_module().check_binding(binding)
    digest = row['trial_binding_sha256']
    require(binding['trial_binding_sha256'] == plan['trial_binding_sha256'] == declared['trial_binding_sha256'] == digest,
            'Canonical trial binding mismatch')
    require(row['commands_sha256'] == declared['commands_sha256'] == plan['commands_sha256']
            == evaluation['commands_sha256'] == sha(command_path), 'Original command SHA mismatch')
    require(sha(plan_path) == evaluation['plan_sha256'], 'Native plan SHA mismatch')
    require(row['gesture_id'] == plan['gesture_id'] == binding['multifinger_pose_finger_type']
            and row['extra_depth_index'] == binding['multifinger_pose_depth_type']
            and row['object_id'] == Path(plan['arguments']['object_root']).name,
            'Trial object/gesture/depth binding mismatch')
    lane = int(execution.name.removeprefix('execution_'))
    staged = cohort_dir/'plans'/('p%03d' % lane)/'commands.npz'
    require(sha(staged) == row['commands_sha256'], 'Staged commands changed')
    with np.load(command_path, allow_pickle=False) as commands, np.load(trace_path, allow_pickle=False) as trace:
        contract = command_contract(plan, commands)
        n = row['physics_steps']; total = contract['expected_command_steps']
        require(type(n) is int and 0 < n < total and evaluation['total_physics_steps'] == n
                and evaluation['declared_command_steps'] == total
                and row['expected_command_steps'] == total, 'Trace length is not a declared proper prefix')
        require(str(commands['trial_binding_sha256'].item()) == digest, 'Commands do not echo binding')
        require(len(trace['phase']) == n, 'Truncated or extra trace rows')
        names = list(map(str, trace['joint_names']))
        cols = [names.index(str(name)) for name in commands['joint_names']]
        require(np.array_equal(trace['joint_target'][:, cols], commands['joint_target'][:n].astype(np.float32)),
                'Actual command target prefix mismatch')
        require(np.array_equal(trace['phase'], commands['phase'][:n]), 'Actual phase prefix mismatch')
        require(np.array_equal(trace['desired_palm'], commands['desired_palm'][:n]), 'Actual palm command prefix mismatch')
        require(np.array_equal(trace['time_s'], commands['time_s'][:n])
                and float(evaluation['dt_s']) == contract['original_dt_s'], 'Actual clock prefix mismatch')
        for name in ('joint_pos','joint_target','object_pose_xyzw','desired_palm','object_depth_m','robot_depth_m'):
            require(len(trace[name]) == n and np.isfinite(trace[name]).all(), 'Invalid trace array: ' + name)
        phase = trace['phase'].astype(str)
        depth = np.maximum(trace['object_depth_m'], trace['robot_depth_m'])
        trigger = np.where(depth > .005, 'deep_collision_stop', '')
        settle_indices = np.flatnonzero(phase == 'settle')
        drift = np.zeros(n)
        if len(settle_indices) and phase[-1] != 'settle':
            last_settle = settle_indices[-1]
            settled = trace['object_pose_xyzw'][last_settle]
            require(np.array_equal(settled, np.asarray(evaluation['settled_object_pose_xyzw'],dtype=settled.dtype)),
                    'Declared settled reference differs from trace')
            drift = np.linalg.norm(trace['object_pose_xyzw'][:, :2].astype(np.float64)-settled[:2].astype(np.float64), axis=1)
            moving = np.isin(phase, ('home_to_pregrasp','approach'))
            # Native source overwrites deep collision reason with drift reason.
            trigger = np.where(moving & (drift > .03), 'object_pushed_before_closure', trigger)
        hits = np.flatnonzero(trigger != '')
        require(len(hits) == 1 and int(hits[0]) == n-1 and trigger[-1] == reason,
                'Native first stop predicate does not match trace endpoint')
        last_phase = str(phase[-1])
        last_drift = float(drift[-1]); last_depth = float(depth[-1])
    evidence_paths = [ev_path, trace_path, plan_path, command_path, staged,
                      Path(row['binding_json']), prepared_path, result_path, receipt_path]
    return dict(schema='anydex_native_task_stop_certificate_v1',certified=True,
                outcome='known_task_terminated_null',trial_id=row['trial_id'],object_id=row['object_id'],
                reason=reason,label=None,label_eligible=False,physics_steps=n,
                expected_command_steps=total,last_phase=last_phase,
                last_horizontal_drift_m=last_drift,last_penetration_m=last_depth,
                original_native_gates=evaluation['gates'],worker_returncode=0,
                evidence_sha256={str(path):sha(path) for path in evidence_paths},
                native_source_sha256=evaluation['source_sha256'],
                checked_prefix_fields=['joint_target','phase','desired_palm','time_s'],
                no_execution_or_relabelling=True)
