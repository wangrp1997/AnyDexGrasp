"""Pure Boya pre-head candidate-domain filter; no simulator, models or file I/O.

Input rows are the ORIGINAL parser layout:
  score, width, depth, rotation(9, row-major), camera XYZ = N x 15.
N x 16 is also accepted with one unused trailing metadata column.  N x 17
GraspGroup rows include a different height/depth layout and are rejected.
No DH3 bounds, direction or camera rectangle are defaults for Boya.

Call apply_boya_domain(raw, T_WC, config, source_provenance={...}), then apply
result.keep_indices to raw AND paired features BEFORE CGR top-K/head inference.
The mask preserves original source indices and never canonicalizes rotations.
All three domain predicates use strict inequalities; boundaries are excluded.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping

import numpy as np

RAW_ROTATION_ATOL = 2e-4   # Structural check for original float32 rotations.
CAMERA_RIGID_ATOL = 1e-6


def _array_ref(value: np.ndarray) -> dict[str, Any]:
    a = np.ascontiguousarray(value)
    return dict(shape=list(a.shape), dtype=a.dtype.str,
                c_order_bytes_sha256=sha256(a.tobytes()).hexdigest())


def _json_copy(value: Any, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{name} must contain finite JSON values') from exc


def _numeric_array(value: Any, name: str, shape: tuple | None = None) -> np.ndarray:
    a = np.asarray(value)
    if a.dtype.kind not in 'fi' or not np.isfinite(a).all():
        raise ValueError(f'{name} must contain finite real numbers')
    if shape is not None and a.shape != shape:
        raise ValueError(f'{name} must have shape {shape}, got {a.shape}')
    return a


def _raw_rows(raw: Any, *, rigid: bool) -> np.ndarray:
    a = _numeric_array(raw, 'raw parser candidates')
    if a.ndim != 2 or a.shape[1] not in (15, 16):
        raise ValueError('Raw parser rows must be Nx15 or Nx16 with trailing metadata; not GraspGroup Nx17')
    if rigid:
        R = a[:, 3:12].astype(np.float64).reshape(-1, 3, 3)
        if (np.any(np.abs(R.transpose(0, 2, 1) @ R - np.eye(3)) > RAW_ROTATION_ATOL)
                or np.any(np.abs(np.linalg.det(R) - 1.) > RAW_ROTATION_ATOL)):
            raise ValueError('Raw parser rotation must be proper SO(3), within declared float32 tolerance')
        if np.any(a[:, 2] < 0):
            raise ValueError('Raw parser depth must be nonnegative')
    return a


def _camera_transform(value: Any) -> np.ndarray:
    T = _numeric_array(value, 'T_WC', (4, 4)).astype(np.float64)
    if not np.allclose(T[3], [0., 0., 0., 1.], atol=1e-8, rtol=0):
        raise ValueError('T_WC homogeneous last row is invalid')
    R = T[:3, :3]
    if (not np.allclose(R.T @ R, np.eye(3), atol=CAMERA_RIGID_ATOL, rtol=0)
            or abs(np.linalg.det(R) - 1.) > CAMERA_RIGID_ATOL):
        raise ValueError('T_WC rotation must be proper SO(3)')
    return T


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Require all Boya domain values explicitly; reject missing/unknown keys."""
    if not isinstance(config, Mapping):
        raise ValueError('Explicit Boya domain config mapping required')
    required = {'world_workspace_aabb_m', 'approach_world_cone', 'raw_width_bounds_m'}
    if set(config) != required:
        raise ValueError(f'Config requires exactly {sorted(required)}')
    bounds = _numeric_array(config['world_workspace_aabb_m'], 'world workspace AABB', (2, 3)).astype(float)
    if np.any(bounds[0] >= bounds[1]):
        raise ValueError('Each workspace minimum must be strictly below its maximum')
    width = _numeric_array(config['raw_width_bounds_m'], 'raw width bounds', (2,)).astype(float)
    if not (0 <= width[0] < width[1]):
        raise ValueError('Explicit raw width bounds require 0 <= minimum < maximum')
    cone = config['approach_world_cone']
    if not isinstance(cone, Mapping) or set(cone) != {'axis_world', 'cos_threshold'}:
        raise ValueError('World cone requires exactly axis_world and cos_threshold')
    axis = _numeric_array(cone['axis_world'], 'world cone unit axis', (3,)).astype(float)
    if abs(np.linalg.norm(axis) - 1.) > 1e-8:
        raise ValueError('World cone axis must already be unit length; no silent normalization')
    threshold = _numeric_array(cone['cos_threshold'], 'cone cosine threshold', ())
    threshold = float(threshold)
    if not -1. <= threshold <= 1.:
        raise ValueError('Cone cosine threshold must lie in [-1,1]')
    return dict(world_workspace_aabb_m=bounds.tolist(),
                approach_world_cone=dict(axis_world=axis.tolist(), cos_threshold=threshold),
                raw_width_bounds_m=width.tolist())


@dataclass(frozen=True)
class DomainResult:
    keep_indices: np.ndarray
    combined_mask: np.ndarray
    gate_masks: dict[str, np.ndarray]
    world_points_m: np.ndarray
    world_approach_axes: np.ndarray
    receipt: dict[str, Any]


def apply_boya_domain(raw: Any, T_WC: Any, config: Mapping[str, Any], *,
                      source_provenance: Mapping[str, Any]) -> DomainResult:
    """Filter paired original raw rows; source files must be bound by caller.

    File provenance is copied, not independently read/verified by this pure
    helper.  Actual raw-array, transform, policy and mask bytes are hashed here.
    Supply the caller's frozen observation/NPZ path+SHA refs as provenance.
    """
    a = _raw_rows(raw, rigid=True)
    T = _camera_transform(T_WC)
    c = validate_config(config)
    if not isinstance(source_provenance, Mapping) or not source_provenance:
        raise ValueError('Nonempty explicit input source provenance required')
    provenance = _json_copy(dict(source_provenance), 'source_provenance')

    R_CG = a[:, 3:12].astype(np.float64).reshape(-1, 3, 3)
    # CGR +X is approach. Translation affects XYZ, NEVER a direction vector.
    approaches = R_CG[:, :, 0] @ T[:3, :3].T
    points = a[:, 12:15].astype(np.float64) @ T[:3, :3].T + T[:3, 3]
    lo, hi = np.asarray(c['world_workspace_aabb_m'])
    wlo, whi = c['raw_width_bounds_m']
    cone = c['approach_world_cone']
    widths = (a[:, 1] > wlo) & (a[:, 1] < whi)
    workspace = np.all((points > lo) & (points < hi), axis=1)
    approach = (approaches @ np.asarray(cone['axis_world'])) > cone['cos_threshold']
    gates = dict(raw_width=widths, world_workspace=workspace, world_approach=approach)
    combined = widths & workspace & approach
    indices = np.flatnonzero(combined).astype(np.int64)
    policy_bytes = json.dumps(c, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    receipt = dict(
        schema='anydex_boya_prehead_domain_v1',
        scope='Boya port input-domain filter, before CGR top-K/heads; not full geometry or arm admission',
        source_provenance=provenance,
        source_file_verification='Caller-supplied refs copied; no file I/O in this pure helper',
        raw_input=_array_ref(a), input_T_WC=_array_ref(np.asarray(T_WC)), T_WC=T.tolist(),
        config=c, config_sha256=sha256(policy_bytes).hexdigest(),
        raw_layout='score,width,depth,R9,XYZ' + (',unused_tail' if a.shape[1] == 16 else ''),
        frame_semantics=dict(input='camera optical, original recovered parser frame',
                             output='world', approach='R_WC @ R_CG[:,0]',
                             point='R_WC @ XYZ_C + t_WC'),
        comparisons='strict open bounds: width_min<w<width_max; AABB_min<XYZ_W<AABB_max; axis_W dot approach_W>cos_threshold',
        validation_tolerances=dict(raw_rotation_atol=RAW_ROTATION_ATOL,
                                   camera_rotation_atol=CAMERA_RIGID_ATOL),
        counts=dict(input=len(a), **{f'{name}_pass': int(mask.sum()) for name, mask in gates.items()},
                    after_width=int(widths.sum()), after_width_workspace=int((widths & workspace).sum()),
                    joint_pass=int(combined.sum())),
        kept_source_indices=indices.tolist(), kept_indices=_array_ref(indices),
        joint_mask=_array_ref(combined),
        gate_masks={name: _array_ref(mask) for name, mask in gates.items()},
        no_physics=True, no_head_inference=True)
    return DomainResult(indices, combined, gates, points, approaches, receipt)


# Descriptive alias for callers that prefer a mask-named entry point.
boya_domain_mask = apply_boya_domain


def released_dh3_camera_mask(raw: Any, minwidth: float, maxwidth: float) -> np.ndarray:
    """Source-test reference ONLY: unchanged three DH3 camera predicates.

    This author rectangle is never used by apply_boya_domain.  Widths must be
    explicit because the official collect and deploy lower bounds differ.
    Rotation/XYZ stay in original author camera frame for this reference.
    """
    a = _raw_rows(raw, rigid=False)
    bounds = _numeric_array([minwidth, maxwidth], 'DH3 reference widths', (2,))
    if not 0 <= bounds[0] < bounds[1]:
        raise ValueError('Explicit finite DH3 reference width bounds required')
    mask = (a[:, 9] > .9) & (a[:, 1] < maxwidth) & (a[:, 1] > minwidth)
    workspace = ((a[:, 12] > -.25) & (a[:, 12] < .25)
                 & (a[:, 13] > -.205) & (a[:, 13] < .03))
    return workspace & mask
