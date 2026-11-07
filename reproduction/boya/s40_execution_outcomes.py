"""Pure collection-ledger reconciliation; no robot, Torch or simulator imports.

Artifact verification is explicit: validate_full_label(row) must verify the full
bound execution; certify_terminal(row) must be the source-bound native terminal
certificate. No artifact-free early stop inference or null -> 0 conversion.
proven_unexecuted_ids is supplied only from an immutable dispatcher launch ledger
that confirms a process/wave was never submitted; missing files are insufficient.
"""
from __future__ import annotations
KNOWN_TERMINAL_REASONS = frozenset(('object_pushed_before_closure','deep_collision_stop'))

def _exact_int(value):return type(value) is int

def partition_execution_rows(rows,expected_ids,*,validate_full_label,certify_terminal,proven_unexecuted_ids=()):
    expected=list(expected_ids);proof=set(proven_unexecuted_ids)
    if len(expected)!=len(set(expected)) or any(not isinstance(i,str) or not i for i in expected):
        raise ValueError('Expected nonempty unique string identities')
    if proof-set(expected):raise ValueError('Unexecuted proof has foreign identities')
    ids=[r.get('trial_id')for r in rows]
    if len(ids)!=len(set(ids))or set(ids)-set(expected):raise ValueError('Duplicate or foreign executor rows')
    byid=dict(zip(ids,rows));eligible=[];terminal=[];unexecuted=[];unknown=[];certificates={};known_steps=0
    for tid in expected:
        row=byid.get(tid)
        if row is None:
            (unexecuted if tid in proof else unknown).append(dict(trial_id=tid,reason='proven_not_dispatched'if tid in proof else'missing_row_after_submission_or_without_dispatch_proof'))
            continue
        n=row.get('physics_steps');label=row.get('label');is_steps=_exact_int(n)and n>=0
        if is_steps:known_steps+=n
        if tid in proof:
            if not is_steps or n!=0 or label is not None or row.get('label_eligible')is not False:
                raise ValueError('Unexecuted proof conflicts with saved execution: '+tid)
            unexecuted.append(dict(trial_id=tid,reason='proven_not_dispatched'));continue
        if not is_steps:
            unknown.append(dict(trial_id=tid,reason='missing_or_invalid_actual_physics_step_count'));continue
        # bool is not accepted as a binary JSON training label despite bool == 1.
        if _exact_int(label)and label in(0,1)and row.get('label_eligible')is True:
            if n<=0:
                unknown.append(dict(trial_id=tid,reason='binary_label_has_no_actual_steps'));continue
            try:
                validate_full_label(row)
                eligible.append(row)
            except Exception as error:
                unknown.append(dict(trial_id=tid,reason='full_label_artifact_verification_failed',error=str(error)))
            continue
        if label is None and row.get('label_eligible')is False and n>0:
            try:
                c=certify_terminal(row)
                expected_steps=c.get('expected_command_steps')
                if not(c.get('certified')is True and c.get('label')is None and c.get('label_eligible')is False
                    and c.get('outcome')=='known_task_terminated_null' and c.get('trial_id')==tid
                    and c.get('object_id')==row.get('object_id')and c.get('physics_steps')==n
                    and _exact_int(expected_steps)and 0<n<expected_steps and c.get('reason')in KNOWN_TERMINAL_REASONS):
                    raise ValueError('Malformed or foreign source-bound terminal certificate')
                terminal.append(row);certificates[tid]=c
            except Exception as error:
                unknown.append(dict(trial_id=tid,reason='uncertified_partial_or_task_execution',error=str(error)))
            continue
        unknown.append(dict(trial_id=tid,reason='zero_step_without_never_dispatched_proof'if n==0 else'malformed_or_ineligible_binary_outcome'))
    consumed={r['trial_id']for r in eligible+terminal};unknown_ids={r['trial_id']for r in unknown}
    return dict(eligible_rows=eligible,terminal_null_rows=terminal,terminal_certificates=certificates,
        proven_unexecuted_rows=unexecuted,unknown_rows=unknown,consumed_trial_ids=sorted(consumed),
        # Unknown may have executed. Never replay it automatically.
        automatic_replay_forbidden_trial_ids=sorted(consumed|unknown_ids),
        proven_unexecuted_trial_ids=[r['trial_id']for r in unexecuted],
        should_stop_new_dispatch=bool(unknown),complete=not unknown and not unexecuted,
        known_recorded_physics_steps=known_steps,physical_cost_complete=not unknown,
        actual_grasps_with_positive_recorded_steps=sum(_exact_int(r.get('physics_steps'))and r['physics_steps']>0 for r in rows),
        eligible_labels=len(eligible),positives=sum(r['label']==1 for r in eligible),negatives=sum(r['label']==0 for r in eligible),
        terminal_nulls=len(terminal),proven_unexecuted=len(unexecuted),unknown=len(unknown),
        new_physics_steps=0,new_training_updates=0)


def inspect_execution(doc, expected_ids, returncode):
    """Verify final native artifacts before admitting any object as complete."""
    if (returncode != 0 or doc.get('status') not in ('complete','complete_with_null')
        or doc.get('error') or doc.get('infrastructure_error')
        or doc.get('input_sha256_unchanged') is not True
        or doc.get('source_sha256_unchanged') is not True):
        raise ValueError('Executor did not close with intact source/input evidence')
    rows=doc.get('trials',[])
    if (type(doc.get('physical_steps')) is not int
        or any(type(r.get('physics_steps')) is not int for r in rows)
        or doc['physical_steps'] != sum(r['physics_steps']for r in rows)):
        raise ValueError('Executor row physics ledger differs from aggregate')
    cohorts=doc.get('cohorts',[])
    if not cohorts or sum(c.get('environment_steps',-1)for c in cohorts)!=doc['physical_steps']:
        raise ValueError('Serial cohort clocks differ from recorded physical cost')
    from resume_collection_execution_v4 import inspect_rows
    from terminal_collection_outcome import certify_known_task_stop
    def validate_full(row):
        if type(row.get('label')) is not int or row['label'] not in (0,1):
            raise ValueError('Exact binary JSON integer required')
        inspect_rows([row],[row['trial_id']])
    result=partition_execution_rows(doc.get('trials',[]),expected_ids,
        validate_full_label=validate_full,certify_terminal=certify_known_task_stop)
    if not result['complete'] or result['should_stop_new_dispatch']:
        raise ValueError('Unknown or incomplete execution: '+str(result['unknown_rows']))
    return result
