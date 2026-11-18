"""The original 96 full-source camera rays using the validated Numba scanner."""
import numpy as np
from author_fast_geometry_checks import first_depth


def ray_diagnostic(mesh_world,depth,K,T,seed=20261105):
    vertices=(mesh_world.vertices-T[:3,3])@T[:3,:3]
    if np.any(vertices[:,2]<=0):raise ValueError('Object behind camera')
    uv=vertices[:,:2]/vertices[:,2:3]@K[:2,:2].T+K[:2,2]
    lo=np.maximum(np.floor(uv.min(0)).astype(int),[0,0])
    hi=np.minimum(np.ceil(uv.max(0)).astype(int),[depth.shape[1]-1,depth.shape[0]-1])
    if (hi<=lo).any():raise ValueError('Object projection outside camera')
    rng=np.random.default_rng(seed)
    pixels=np.column_stack([rng.integers(lo[0],hi[0]+1,96),rng.integers(lo[1],hi[1]+1,96)])
    rays=np.c_[(pixels+.5-K[:2,2])/np.diag(K)[:2],np.ones(96)]@T[:3,:3].T
    triangles=np.asarray(mesh_world.triangles)
    answers=[first_depth(triangles,T[:3,3],ray)for ray in rays]
    expected=np.array([a[0]for a in answers]);is_object=np.array([a[1]for a in answers])
    observed=depth[pixels[:,1],pixels[:,0]]
    valid=np.isfinite(expected)&np.isfinite(observed)&(expected>.1)&(expected<1.)&(observed>.1)&(observed<1.)
    diff=np.abs(expected[valid]-observed[valid])
    if not len(diff):raise ValueError('No valid diagnostic rays')
    report=dict(ray_count=len(rays),valid_rays=int(valid.sum()),source_object_first_hits=int(is_object.sum()),
        pixel_center_offset=.5,median_absolute_depth_error_m=float(np.median(diff)),
        p90_absolute_depth_error_m=float(np.quantile(diff,.9)),max_absolute_depth_error_m=float(diff.max()),
        pixels_xy=pixels.tolist(),expected_source_first_depth_m=[float(x)if np.isfinite(x)else None for x in expected],
        observed_rtx_depth_m=[float(x)if np.isfinite(x)else None for x in observed],source_first_hit_is_object=is_object.tolist(),
        boundary='Full source triangles plus analytic z=0 plane; original pixel centers and diagnostic gates. No sampling or physics changes.',
        full_source_triangles=len(triangles),implementation='author_fast_geometry_checks.first_depth; fastmath=False')
    report['diagnostic_pass']=bool(is_object.sum()>=2 and report['median_absolute_depth_error_m']<.002 and report['p90_absolute_depth_error_m']<.004)
    return report
