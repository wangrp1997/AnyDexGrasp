"""User-approved object/object penetration diagnostics, with real lift/load labels.

Original numeric gates and source lift outputs remain visible. This changes
admission only; it never changes simulation parameters or contact data.
"""
import copy
import math

import author_mujoco_control as C

SCHEMA='anydex_boya_object_overlap_diagnostic_lift_v1'
POLICY=dict(schema=SCHEMA,object_object_penetration='diagnostic_only',
    remaining_penetration_limit_m=C.LIFT.POLICY['max_penetration_m'],
    early_stop_penetration_m=C.LIFT.POLICY['early_stop_penetration_m'],
    lift_policy=copy.deepcopy(C.LIFT.POLICY),other_numeric_gates='original',
    authorization='User: object penetration can be tolerated; require an actual grasp lift')


def penetration_summary(rows,object_ids):
    paths={'/World/'+key for key in object_ids}
    if len(paths)!=len(object_ids) or not paths:raise ValueError('Unique original object identities required')
    result=dict(all_m=0.,object_object_m=0.,remaining_m=0.)
    for row in rows:
        for contact in row['contacts']:
            bodies=contact['bodies'];separation=float(contact['separation_m'])
            if len(bodies)!=2 or not all(isinstance(b,str) for b in bodies) or not math.isfinite(separation):
                raise ValueError('Finite separation and two named bodies required')
            depth=max(0.,-separation)
            category='object_object_m' if bodies[0]!=bodies[1] and all(b in paths for b in bodies) else 'remaining_m'
            result[category]=max(result[category],depth);result['all_m']=max(result['all_m'],depth)
    return result


def evaluate(rows,object_ids,maxima,expected_steps,*,stop=None):
    if not rows:raise ValueError('A recorded grasp trace is required')
    depths=penetration_summary(rows,object_ids)
    if abs(depths['all_m']-maxima['penetration_m'])>1e-12:raise ValueError('Original penetration maximum differs from full contacts')
    original=C.source_gates(maxima,len(rows),expected_steps)
    adjusted=dict(maxima,penetration_m=depths['remaining_m'])
    applied=C.source_gates(adjusted,len(rows),expected_steps)
    lift=C.LIFT.evaluate_lift(rows,object_ids,.004)
    # Preserve the pre-existing runtime's 5mm deep-collision stop. A missed
    # stop in a supplied trace cannot grant admission under this policy.
    deep_stop=depths['all_m']>POLICY['early_stop_penetration_m']
    admission=stop is None and not deep_stop and all(applied.values())
    return dict(schema=SCHEMA,policy=copy.deepcopy(POLICY),original_gates=original,
        acceptance_gates=applied,penetration=depths,original_stop=stop,
        original_deep_collision_stop_triggered=deep_stop,policy_admission_pass=admission,
        observed_lift_task_success=lift['task_success'],
        diagnostic_task_success=admission and lift['task_success'],evaluation=lift,
        physics_parameters_changed=False,original_contact_fields_dropped=False)
