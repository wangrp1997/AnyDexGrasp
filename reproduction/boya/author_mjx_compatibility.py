"""Explicit numerical compatibility for a GPU-only copy of a MuJoCo model."""
import math


def apply_gpu_ccd_tolerance(model):
    """Use at least the native default CCD tolerance for float32 MJX-Warp.

    The source hand's 1e-10 m tolerance loses real contact pairs on this GPU
    backend. A fixed diagnostic recovered those pairs at the MuJoCo default
    1e-6 m. This is collision-algorithm convergence precision, not a penetration
    acceptance threshold. Call on a dedicated GPU model; keep the CPU reference.
    """
    import mujoco
    before=float(model.opt.ccd_tolerance)
    if not math.isfinite(before) or before<0:raise ValueError('Finite nonnegative CCD tolerance required')
    default=float(mujoco.MjOption().ccd_tolerance)
    after=max(before,default)
    model.opt.ccd_tolerance=after
    return dict(schema='anydex_mjx_float32_ccd_compatibility_v1',
        source_ccd_tolerance_m=before,gpu_ccd_tolerance_m=after,
        native_default_ccd_tolerance_m=default,backend='mjx-warp',
        changed=after!=before,penetration_acceptance_threshold_changed=False,
        solver_tolerance_changed=False,geometry_changed=False,control_changed=False)
