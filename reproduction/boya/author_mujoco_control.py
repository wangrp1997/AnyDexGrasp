"""Explicit MuJoCo diagnostic controls/observations; no production label writer.

Keep the original 24 position commands, six source-derived velocity targets,
thumb effort override and source hand excludes. Engine/contact differences
remain explicit. Older frozen preflights are never modified by this module.
"""
import ast
from functools import lru_cache
from pathlib import Path

import numpy as np
import author_mujoco_source_filters as FILTERS
import author_lift as LIFT

S = FILTERS.S
R, B, T = S.ROBOT, S.B, S.T
EXECUTOR = T.CANON/'scripts/research/execute_template_arm_positive.py'


def source_velocity_targets(targets, names, arm_names):
    tree = ast.parse(EXECUTOR.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'planned_arm_velocity_targets')
    scope = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(EXECUTOR), 'exec'), scope)
    return scope[fn.name](targets, .004, names, arm_names)


@lru_cache(maxsize=1)
def gate_expression():
    tree = ast.parse(EXECUTOR.read_text())
    expression = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(x, ast.Name) and x.id == 'gates' for x in n.targets)
                      and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                      and n.value.func.id == 'dict' and any(k.arg == 'joint_limits' for k in n.value.keywords))
    return compile(ast.Expression(expression), str(EXECUTOR), 'eval')


def source_gates(maxima, step, expected):
    """Evaluate the actual frozen native executor's seven numeric expressions."""
    values = eval(gate_expression(),
                  dict(np=np, maxima=maxima, step=step, target_array=range(expected)))
    if set(values) != set(LIFT.ORIGINAL_GATES):
        raise ValueError('Original gate schema changed')
    return {k: bool(v) for k, v in values.items()}


def add_velocity_drives(spec, parameters, arm_names):
    """Sum kp*(qdes-q)+kd*(vdes-v), then cap total joint force once."""
    motors = {a.name: a for a in spec.actuators}
    import mujoco
    for name in arm_names:
        motors[name].forcelimited = 0
        spec.add_actuator(name='velocity_'+name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[parameters[name]['damping']]+[0.]*9,
            biastype=mujoco.mjtBias.mjBIAS_NONE, ctrllimited=0, forcelimited=0)


def controls(model, data, names, target, velocity, arm_names):
    import mujoco
    if model.nu != len(names)+len(arm_names):
        raise ValueError('Every position and derived velocity motor required')
    for labels, values in ((names, target), (['velocity_'+n for n in arm_names],
                            [velocity[names.index(n)] for n in arm_names])):
        values = np.asarray(values)
        if values.shape != (len(labels),) or not np.isfinite(values).all():
            raise ValueError('Finite complete original controls required')
        indices = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in labels]
        if min(indices) < 0:
            raise ValueError('Missing named actuator')
        data.ctrl[indices] = values
        if not np.array_equal(data.ctrl[indices], values):
            raise ValueError('Original command changed')


def build(plan, names, scene):
    import mujoco
    spec, _, planning, reference, parameters, _ = R.build_robot(plan, names)
    from canondex.grasp.template_gripper_drive import thumb_root_effort_overrides
    overrides = thumb_root_effort_overrides(B.read(plan['arguments']['diagnostic_config'])['thumb_root_effort_limit_nm'])
    joints = {j.name: j for j in spec.joints}
    actuators = {a.name: a for a in spec.actuators}
    for name, effort in overrides.items():
        parameters[name]['joint_effort_limit'] = effort
        joints[name].actfrcrange = [-effort, effort]
        actuators[name].forcerange = [-effort, effort]
    arm_names = list(reference.arm_home)
    add_velocity_drives(spec, parameters, arm_names)
    excludes = FILTERS.apply_filters(spec, plan['arguments']['hand_scene'])
    objects = S.add_scene(spec, scene)
    for geom in spec.geoms:
        geom.friction = [.6, .02, .0001]
        geom.condim = 4
    model = spec.compile()
    if (model.nq, model.nv, model.nu, model.neq) != (49, 46, 30, 4):
        raise ValueError('Original robot/three-object dimensions changed')
    return spec, model, planning, reference, parameters, arm_names, excludes, objects


def pose_matrix(model, data, body):
    result = np.eye(4)
    result[:3, :3] = data.xmat[body].reshape(3, 3)
    result[:3, 3] = data.xpos[body]
    return result


def error(actual, expected):
    delta = np.linalg.inv(expected)@actual
    return float(np.linalg.norm(delta[:3, 3])), float(np.arccos(np.clip((np.trace(delta[:3, :3])-1)/2, -1, 1)))


def body_mapping(model, planning, ids):
    import mujoco
    result = {}
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name in planning.fk.links:
            owner = planning.fk.fixed_attachment(name, np.eye(4))[0]
            result[i] = '/World/Robot/'+owner
        elif name in ids:
            result[i] = '/World/'+name
        elif i == 0:
            result[i] = '/World/ground'
        else:
            raise ValueError('Unmapped physical body: '+str(name))
    return result


def contact_rows(model, data, mapping):
    """Every active mjContact once; canonical object-first views are derived.

    Native frame's first axis points geom1 -> geom2. Keep native wrench/frame;
    no claim of PhysX impulse or filtered-array compatibility is made.
    """
    import mujoco
    rows = []
    for i in range(data.ncon):
        c = data.contact[i]
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, wrench)
        geoms = [int(c.geom1), int(c.geom2)]
        bodies = [int(model.geom_bodyid[g]) for g in geoms]
        rows.append(dict(native_index=i, geom_ids=geoms, body_ids=bodies,
            bodies=[mapping[b] for b in bodies], separation_m=float(c.dist),
            position_world_m=c.pos.tolist(), contact_frame_world=c.frame.tolist(),
            wrench_contact_frame=wrench.tolist(), normal_force_n=float(wrench[0]),
            constraint_dimension=int(c.dim), constraint_address=int(c.efc_address),
            friction=c.friction.tolist(), solref=c.solref.tolist(), solimp=c.solimp.tolist(),
            exclude=int(c.exclude), inclusion_margin_m=float(c.includemargin)))
    return rows


def object_views(contacts, ids):
    views = []
    for c in contacts:
        for key in ids:
            path = '/World/'+key
            if path not in c['bodies']:
                continue
            index = c['bodies'].index(path)
            views.append(dict(native_index=c['native_index'], bodies=[path, c['bodies'][1-index]],
                              normal_force_n=c['normal_force_n'], separation_m=c['separation_m']))
    return views


def numeric_maxima(rows, planning, names, desired, reference, legal_hand, ids):
    """Recompute numeric gates from saved states/points, independent of simulator."""
    m = dict(joint_limit_violation_rad=0., mimic_error_rad=0., penetration_m=0.,
        palm_position_error_m=0., palm_orientation_error_rad=0., illegal_object_contact_force_n=0.,
        base_position_drift_m=0., base_orientation_drift_rad=0.)
    first = {}
    robot_paths = sorted({'/World/Robot/'+planning.fk.fixed_attachment(n, np.eye(4))[0] for n in planning.fk.links})
    for row, wanted in zip(rows, desired):
        q = dict(zip(names, row['joint_position']))
        transforms = planning.fk.transforms(q, reference.base_transform)
        pos, angle = error(transforms['palm'], wanted)
        values = dict(joint_limit_violation_rad=max(max(float(j.find('limit').get('lower'))-q[n],
                    q[n]-float(j.find('limit').get('upper')), 0.) for n, j in planning.joints.items()),
            mimic_error_rad=max(abs(q[c]-q[d]*k-o) for c, (d, k, o) in planning.mimics.items()),
            penetration_m=max([0.]+[-c['separation_m'] for c in row['contacts']]),
            palm_position_error_m=pos, palm_orientation_error_rad=angle,
            base_position_drift_m=error(np.asarray(row['base_transform']), reference.base_transform)[0],
            base_orientation_drift_rad=error(np.asarray(row['base_transform']), reference.base_transform)[1])
        views = object_views(row['contacts'], ids)
        values['illegal_object_contact_force_n'] = max(T.object_robot_loads('/World/'+key, views, robot_paths, legal_hand)
                                                      ['illegal_robot_support_force_n'] for key in ids)
        for k, value in values.items():
            m[k] = max(m[k], float(value))
        gates = source_gates(m, row['step'], len(rows))
        for gate, passed in gates.items():
            if not passed and gate != 'complete_execution' and gate not in first:
                first[gate] = dict(step=row['step'], phase=row['phase'])
    return m, first
