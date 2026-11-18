"""Explicit source mass properties and already-declared table rolling friction.

Source PhysX uses the original mesh COM/inertia, not the overlapping CoACD
pieces' compound estimate. Table-only six-dimensional pairs activate the
existing 0.1mm rolling coefficient; hand/object contact remains unchanged.
This is a new dynamics domain, not a claim of cross-engine equivalence.
"""
import numpy as np

SCHEMA='anydex_boya_source_inertia_table_rolling_v1'


def mass_properties(asset,mass=.1):
    mesh=asset.mesh
    matrix=np.asarray(mesh.moment_inertia,dtype=float)*(mass/float(mesh.mass))
    values,axes=np.linalg.eigh(matrix)
    if not np.isfinite(values).all() or min(values)<=0:raise ValueError('Positive source mesh inertia required')
    if np.linalg.det(axes)<0:axes[:,0]*=-1
    center=np.asarray(asset.com_local,dtype=float)
    if center.shape!=(3,) or not np.isfinite(center).all():raise ValueError('Finite original source mesh COM required')
    return mass,center,values,axes,matrix


def apply(spec,assets):
    import mujoco
    bodies={b.name:b for b in spec.bodies};rows=[]
    floor=next(g for g in spec.geoms if g.name=='floor')
    for key,asset in assets.items():
        body=bodies[key];mass,center,values,axes,matrix=mass_properties(asset)
        quat=np.zeros(4);mujoco.mju_mat2Quat(quat,axes.ravel())
        body.explicitinertial=True;body.mass=mass;body.ipos=center;body.iquat=quat;body.inertia=values
        rows.append(dict(object_id=key,mass_kg=mass,center_of_mass_m=center.tolist(),
            principal_inertia_kg_m2=values.tolist(),principal_axes=axes.tolist(),inertia_matrix_kg_m2=matrix.tolist()))
    pairs=[]
    for geom in spec.geoms:
        if geom.parent.name not in assets or not geom.contype or not geom.conaffinity:continue
        if not geom.name:raise ValueError('Named object collider required for explicit table pair')
        if not np.array_equal(geom.friction,floor.friction) or not np.array_equal(geom.solimp,floor.solimp):
            raise ValueError('This profile requires the existing equal table/object friction and impedance')
        if any(x<=0 for x in np.r_[geom.solref,floor.solref]):raise ValueError('Positive timeconstant solver format required')
        # Equal priority and solmix=1 in the source candidate. Preserve the
        # effective old pair's solver/friction values, changing only dimension.
        if geom.priority!=floor.priority or geom.solmix!=floor.solmix:raise ValueError('Unmatched contact priority/mix')
        ref=(geom.solref+floor.solref)/2
        friction=[geom.friction[0],geom.friction[0],geom.friction[1],geom.friction[2],geom.friction[2]]
        spec.add_pair(name='table_rolling_'+geom.name,geomname1=floor.name,geomname2=geom.name,
            condim=6,friction=friction,solref=ref,solimp=geom.solimp,
            margin=max(geom.margin,floor.margin),gap=max(geom.gap,floor.gap))
        pairs.append(dict(geom=geom.name,condim_before=4,condim_after=6,friction=friction,solref=ref.tolist()))
    return dict(schema=SCHEMA,objects=rows,table_pairs=pairs,hand_object_contact_dimension_changed=False,
        geometry_changed=False,friction_coefficients_changed=False,rolling_coefficient_now_active=True,
        state_zeroing=False,native_sleep_enabled=False,full_grasp_validation_complete=False)
