"""Opt-in stiff joint stops; URDF ranges and physical drives stay unchanged.

MuJoCo's default 20ms constraint response is compliant. Use the documented
minimum safe positive-format time constant, twice the native timestep, for
every limited hinge. This is a new solver domain, not a URDF correction or a
claim of mathematically hard constraints. No state/target clipping is used.
"""
import math

SCHEMA='anydex_boya_mujoco_stiff_joint_limits_v1'
REFERENCE='https://mujoco.readthedocs.io/en/stable/modeling.html#reference'


def apply(spec):
    import mujoco
    dt=float(spec.option.timestep)
    if not math.isfinite(dt) or not 0<dt<=.004:raise ValueError('Finite native timestep in (0,4ms] required')
    if spec.option.disableflags & (mujoco.mjtDisableBit.mjDSBL_CONSTRAINT|mujoco.mjtDisableBit.mjDSBL_LIMIT):
        raise ValueError('Joint constraints must remain enabled')
    if spec.option.disableflags & mujoco.mjtDisableBit.mjDSBL_REFSAFE:
        raise ValueError('MuJoCo reference safety must remain enabled')
    rows=[]
    for joint in spec.joints:
        if joint.type!=mujoco.mjtJoint.mjJNT_HINGE:continue
        limited=int(joint.limited)
        if limited==0:continue
        # Auto-limits only activates for a nonempty authored range.
        if limited==2 and not joint.range[0]<joint.range[1]:continue
        if not joint.range[0]<joint.range[1]:raise ValueError('Nonempty finite hinge range required')
        before=list(joint.solref_limit)
        joint.solref_limit=[2*dt,1.]
        rows.append(dict(joint=joint.name,range_rad=list(joint.range),
            before_solref=before,after_solref=list(joint.solref_limit),solimp=list(joint.solimp_limit)))
    if not rows:raise ValueError('At least one enabled hinge stop required')
    return dict(schema=SCHEMA,native_dt_s=dt,reference=REFERENCE,joints=rows,
        rule='All limited hinges: timeconstant=2*native_dt, critical damping=1; impedance unchanged',
        ranges_changed=False,drives_changed=False,state_clipping=False,target_clipping=False,
        solver_model_changed=True,full_grasp_validation_complete=False)
